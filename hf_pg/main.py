import os
import glob
import ollama
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from pypdf import PdfReader
import pandas as pd
# 💡 [수정] 허깅페이스 데이터셋 변환을 위한 패키지 임포트 추가
from datasets import Dataset 

# ==========================================
# 0. 설정 상수 정의
# ==========================================
EMBED_MODEL = "bge-m3"
DB_CONN_STR = "dbname=rag_db user=root password=1234 host=localhost port=5432"

# ==========================================
# 1. PDF 로드 및 텍스트 분할 (Text Chunking)
# ==========================================
def extract_and_chunk_pdf(pdf_path, chunk_size=600, chunk_overlap=100):
    print(f"📄 '{pdf_path}' 읽기 시작...")
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += (chunk_size - chunk_overlap)
        
    print(f"✅ PDF 분석 완료: 총 {len(chunks)} 개의 텍스트 조각 생성.")
    return chunks

def extract_and_chunk_excel(excel_path):
    print(f"📊 엑셀 파일 '{excel_path}' 읽기 시작...")
    
    # 💡 [보완] 엑셀 파일을 보다 안정적으로 읽기 위해 openpyxl 엔진을 명시합니다.
    df = pd.read_excel(excel_path, engine='openpyxl')
    df = df.fillna("")
    
    chunks = []
    for index, row in df.iterrows():
        row_text_list = []
        for col in df.columns:
            row_text_list.append(f"{col}: {row[col]}")
        
        chunk_text = ", ".join(row_text_list)
        full_chunk = f"[엑셀 데이터 {index+1}번째 행] {chunk_text}"
        chunks.append(full_chunk)
        
    print(f"✅ 엑셀 분석 완료: 총 {len(chunks)} 개의 데이터 행(청크) 추출 성공.")
    return chunks

# ==========================================
# 2. Ollama 기반 임베딩 추출 함수
# ==========================================
def get_ollama_embedding(text):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response['embedding']

# ==========================================
# 3. PostgreSQL(pgvector) 초기화 및 데이터 저장
# ==========================================
def init_and_save_to_pgvector(chunks):
    conn = psycopg2.connect(DB_CONN_STR)
    cur = conn.cursor()
    
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("DROP TABLE IF EXISTS esg_documents;")
    cur.execute("""
        CREATE TABLE esg_documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(1024)
        );
    """)
    conn.commit()
    
    print("🚀 Ollama 임베딩 연산 및 pgvector 저장 중 (잠시 시간이 소요됩니다)...")
    data_to_insert = []
    for chunk in chunks:
        vector = get_ollama_embedding(chunk)
        data_to_insert.append((chunk, str(vector)))
        
    execute_values(cur, "INSERT INTO esg_documents (content, embedding) VALUES %s", data_to_insert)
    conn.commit()
    
    cur.execute("CREATE INDEX ON esg_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
    conn.commit()
    
    cur.close()
    conn.close()
    print("✅ pgvector에 ESG 지식 데이터베이스 구축 완료!")

# ==========================================
# 4. pgvector 기반 코사인 유사도 검색 알고리즘
# ==========================================
def search_similar_documents(query, top_k=3):
    query_vector = get_ollama_embedding(query)
    
    conn = psycopg2.connect(DB_CONN_STR)
    cur = conn.cursor()
    
    # 💡 [수정] 이중 SELECT 문법 오류 해결 및 올바른 파라미터 바인딩 구조로 변경
    cur.execute("""
        SELECT content 
        FROM esg_documents 
        ORDER BY embedding <=> %s 
        LIMIT %s;
    """, (str(query_vector), top_k))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return [row[0] for row in results]

# ==========================================
# 5. RAG 생성 파트
# ==========================================
def ask_esg_chatbot(model_name, query):
    retrieved_contexts = search_similar_documents(query, top_k=2)
    context = "\n\n".join(retrieved_contexts)
    
    prompt = f"""당신은 ESG 공급망 실사 지침 및 글로벌 규제 전문가입니다. 
주어진 [참고 문서]의 핵심 내용을 기반으로 [사용자 질문]에 신뢰할 수 있는 정보를 제공하세요.
반드시 문서에 나와 있는 내용만을 토대로 요약, 분석해야 합니다.

[참고 문서]
{context}

[사용자 질문]
{query}
"""
    
    print(f"\n🤖 [{model_name}] 모델이 답변을 생성하는 중...")
    response = ollama.generate(model=model_name, prompt=prompt)
    return response['response']

def export_pgvector_to_file_and_hf(repo_id=None):
    conn = psycopg2.connect(DB_CONN_STR)
    cur = conn.cursor()
    
    cur.execute("SELECT id, content, embedding::text FROM esg_documents;")
    rows = cur.fetchall()
    
    cur.close()
    conn.close()
    
    if not rows:
        print("⚠️ DB에 저장된 데이터가 없습니다.")
        return

    df = pd.DataFrame(rows, columns=['id', 'content', 'embedding'])
    df['embedding'] = df['embedding'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])

    df.to_csv("esg_vector_backup.csv", index=False, encoding='utf-8-sig')
    df.to_parquet("esg_vector_backup.parquet", index=False)
    print("💾 로컬 백업 완료: 'esg_vector_backup.csv' 및 '.parquet' 파일이 저장되었습니다.")

    if repo_id:
        print("🤗 허깅페이스 데이터셋 변환 및 업로드 시작...")
        hf_dataset = Dataset.from_pandas(df)
        hf_dataset.push_to_hub(repo_id, private=True)
        print(f"🚀 허깅페이스 저장소({repo_id})에 성공적으로 업로드되었습니다!")


# ==========================================
# 메인 제어 흐름
# ==========================================
if __name__ == "__main__":
    all_chunks = []

    # 1. PDF 폴더 처리
    pdf_folder_path = "./esg_pdf_files"
    if not os.path.exists(pdf_folder_path):
        os.makedirs(pdf_folder_path)
        print(f"📁 '{pdf_folder_path}' 폴더가 생성되었습니다. 분석할 PDF 파일들을 이 폴더에 넣어주세요.")
    
    pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    if pdf_files:
        print(f"📚 {len(pdf_files)}개의 PDF 파일을 찾았습니다. 분석을 시작합니다.")
        for pdf_path in pdf_files:
            try:
                pdf_chunks = extract_and_chunk_pdf(pdf_path)
                all_chunks.extend(pdf_chunks)
            except Exception as e:
                print(f"❌ {pdf_path} 파일 읽기 실패: {e}")
    else:
        print(f"ℹ️ '{pdf_folder_path}' 폴더에 PDF 파일이 없습니다.")

    # 2. 엑셀 폴더 처리
    excel_folder_path = "./esg_excel_files"
    if not os.path.exists(excel_folder_path):
        os.makedirs(excel_folder_path)
        print(f"📁 '{excel_folder_path}' 폴더가 생성되었습니다. 분석할 엑셀 파일들을 이 폴더에 넣어주세요.")
        
    excel_files = glob.glob(os.path.join(excel_folder_path, "*.xlsx")) + glob.glob(os.path.join(excel_folder_path, "*.xls"))
    if excel_files:
        print(f"📊 {len(excel_files)}개의 엑셀 파일을 찾았습니다. 분석을 시작합니다.")
        for excel_path in excel_files:
            try:
                excel_chunks = extract_and_chunk_excel(excel_path)
                all_chunks.extend(excel_chunks)
            except Exception as e:
                print(f"❌ {excel_path} 파일 읽기 실패: {e}")
    else:
        print(f"ℹ️ '{excel_folder_path}' 폴더에 엑셀 파일이 없습니다.")

    # 3. pgvector 통합 저장
    if all_chunks:
        init_and_save_to_pgvector(all_chunks)
    else:
        print("⚠️ 폴더들에 읽어올 파일이 하나도 없어 DB 구축을 건너뜁니다.")

    # 4. RAG 질의응답 및 백업 수행
    if all_chunks:
        test_query = "협력사의 탄소 배출량 실사 의무 규정과 제재 조치는 어떻게 되나요?"
        
        # 💡 [수정] 오타인 qwen3.5 대신 표준 배포 모델인 qwen2.5:latest 로 변경하여 테스트합니다.
        # 실행 전에 반드시 본인 컴퓨터에 해당하는 Ollama 모델이 받아져 있는지 확인하세요.
        try:
            target_model = "qwen3.5:latest" 
            qwen_answer = ask_esg_chatbot(target_model, test_query)
            print(f"\n========= 🌟 {target_model} 답변 =========")
            print(qwen_answer)
        except Exception as e:
            print(f"\n❌ LLM 답변 생성 중 오류 발생: {e}")
            print("💡 힌트: 혹시 Ollama에 해당 LLM 모델이 다운로드(pull)되어 있는지 확인해 보세요.")

        export_pgvector_to_file_and_hf(repo_id=None)