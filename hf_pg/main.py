import os
import sys
import glob
import ollama
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from pypdf import PdfReader
import pandas as pd
from datasets import Dataset 

# ==========================================
# 0-0. Windows 콘솔 인코딩 예외 안전장치 (Global Safe Print)
# ==========================================
def safe_print(*args, **kwargs):
    """
    Windows cp949 콘솔 환경 등에서 유니코드 특수문자나 LLM 답변 특수 외국어(한자 등)가 
    출력될 때 발생하는 UnicodeEncodeError 크래시를 방지하는 범용 출력 함수입니다.
    지원하지 않는 글자는 '?' 등으로 대체하여 출력을 성공시킵니다.
    """
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    file = kwargs.get('file', sys.stdout)
    
    text = sep.join(str(arg) for arg in args)
    try:
        file.write(text + end)
        file.flush()
    except UnicodeEncodeError:
        encoding = getattr(file, 'encoding', 'utf-8') or 'utf-8'
        safe_text = text.encode(encoding, errors='replace').decode(encoding)
        file.write(safe_text + end)
        file.flush()

# 내장 print 함수를 safe_print로 대체하여 인코딩 오류로 인한 크래시 원천 차단
print = safe_print


# ==========================================
# 0. 설정 상수 정의
# ==========================================
EMBED_MODEL = "bge-m3"
DB_CONN_STR = "dbname=rag_db user=root password=1234 host=localhost port=5432"

# ==========================================
# 0-1. 데이터베이스 연결 및 자동 생성 도우미 함수
# ==========================================
def get_db_connection(db_conn_str):
    """
    PostgreSQL 데이터베이스에 연결합니다.
    지정된 데이터베이스('rag_db')가 없을 경우 자동으로 생성합니다.
    서버 연결 자체가 실패할 경우 사용자 친화적인 안내 메시지를 출력합니다.
    """
    try:
        conn = psycopg2.connect(db_conn_str)
        return conn
    except psycopg2.OperationalError as e:
        error_msg = str(e)
        # 데이터베이스가 없어서 발생하는 에러인 경우
        if "does not exist" in error_msg or "database" in error_msg.lower():
            print("[경고] 'rag_db' 데이터베이스가 존재하지 않습니다. 자동 생성을 시도합니다...")
            # postgres 기본 DB에 접속하여 rag_db를 생성합니다.
            postgres_conn_str = db_conn_str.replace("dbname=rag_db", "dbname=postgres")
            try:
                conn_pg = psycopg2.connect(postgres_conn_str)
                conn_pg.autocommit = True
                cur_pg = conn_pg.cursor()
                cur_pg.execute("CREATE DATABASE rag_db;")
                cur_pg.close()
                conn_pg.close()
                print("[성공] 'rag_db' 데이터베이스가 성공적으로 생성되었습니다!")
                
                # 재연결 시도
                return psycopg2.connect(db_conn_str)
            except Exception as create_err:
                print(f"[오류] 데이터베이스 자동 생성 중 오류가 발생했습니다: {create_err}")
                raise e
        else:
            print("\n" + "="*60)
            print("[오류] PostgreSQL 데이터베이스 서버 연결 실패!")
            print(f"상세 에러: {e}")
            print("-"*60)
            print("[팁] 해결 방법:")
            print("1. 로컬 PostgreSQL 서비스가 실행 중인지 확인하세요.")
            print("2. 사용자 ID(root) 및 비밀번호(1234)가 설정 정보와 일치하는지 확인하세요.")
            print("3. Docker를 사용할 경우 다음 명령어로 pgvector를 손쉽게 구동할 수 있습니다:")
            print("   docker run --name pgvector -e POSTGRES_DB=rag_db -e POSTGRES_USER=root -e POSTGRES_PASSWORD=1234 -p 5432:5432 -d pgvector/pgvector:pg16")
            print("="*60 + "\n")
            raise e

# ==========================================
# 0-2. Ollama 모델 관리 및 다운로드 도우미 함수
# ==========================================
def check_and_pull_ollama_model(model_name):
    """
    Ollama에 필요한 모델이 있는지 확인하고, 없을 경우 자동 다운로드(pull)를 시도합니다.
    """
    print(f"[조회] Ollama 모델 '{model_name}' 로컬 설치 상태 검사 중...")
    try:
        models_list = ollama.list()
        downloaded_models = []
        for m in models_list.get('models', []):
            name = m.get('model', m.get('name', ''))
            downloaded_models.append(name)
            # tag가 없는 경우를 대비한 매칭 추가
            if ':' in name:
                downloaded_models.append(name.split(':')[0])
                
        exists = any(model_name == m or model_name in m or m in model_name for m in downloaded_models)
        
        if not exists:
            print(f"[다운로드] 로컬에서 '{model_name}' 모델을 찾을 수 없습니다. 자동 다운로드(pull)를 시작합니다...")
            print("[경고] 모델 크기에 따라 수 분의 시간이 소요될 수 있습니다.")
            ollama.pull(model_name)
            print(f"[성공] 모델 '{model_name}' 다운로드 완료!")
        else:
            print(f"[성공] 모델 '{model_name}' 확인 완료 (사용 가능)")
    except Exception as e:
        print(f"[경고] Ollama 서비스 모델 조회/다운로드 시 실패: {e}")
        print("[팁] Ollama 데스크톱 앱이 실행 중인지 확인하세요. (API 연결 실패)")

def get_embedding_dimension(model_name):
    """
    임베딩 모델에 테스트 문장을 전달하여 반환 벡터의 정확한 차원을 동적으로 감지합니다.
    """
    # 임베딩 모델 사용 가능 여부 선제 체크
    check_and_pull_ollama_model(model_name)
    try:
        test_emb = get_ollama_embedding("test")
        dim = len(test_emb)
        print(f"[크기] 임베딩 모델 '{model_name}' 차원 수 감지 완료: {dim}차원")
        return dim
    except Exception as e:
        print(f"[경고] 임베딩 차원 동적 감지 실패: {e}. 기본값 1024차원으로 설정합니다.")
        return 1024

# ==========================================
# 1. PDF 로드 및 텍스트 분할 (Text Chunking with Metadata)
# ==========================================
def extract_and_chunk_pdf(pdf_path, chunk_size=600, chunk_overlap=150):
    """
    PDF를 읽고 단락 및 문장 경계를 최대한 존중하며 텍스트를 청킹합니다.
    각 청크에는 원본 파일명 및 페이지 정보가 담긴 메타데이터가 보존됩니다.
    """
    print(f"[PDF] '{pdf_path}' 읽기 시작...")
    reader = PdfReader(pdf_path)
    chunks_with_metadata = []
    file_name = os.path.basename(pdf_path)
    
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        text = page.extract_text()
        if not text or not text.strip():
            continue
            
        # 텍스트 간단 정제 (공백 및 줄바꿈 보정)
        text = text.replace('\r\n', '\n').strip()
        
        # 단락별로 구분하여 청킹 (문장 및 문단 단위의 정보 손실 방지)
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 하나의 단락이 설정 크기보다 크면 줄 단위로 분절하여 처리
            if len(para) > chunk_size:
                lines = para.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if len(current_chunk) + len(line) + 1 > chunk_size:
                        if current_chunk:
                            chunks_with_metadata.append({
                                "content": current_chunk.strip(),
                                "source_file": file_name,
                                "source_type": "PDF",
                                "page_or_row": f"{page_num}페이지"
                            })
                        # 중첩 영역 설정
                        overlap_start = max(0, len(current_chunk) - chunk_overlap)
                        current_chunk = current_chunk[overlap_start:] + "\n" + line
                    else:
                        current_chunk += ("\n" if current_chunk else "") + line
            else:
                # 단락을 합쳤을 때 청크 크기를 초과하는 경우
                if len(current_chunk) + len(para) + 2 > chunk_size:
                    if current_chunk:
                        chunks_with_metadata.append({
                            "content": current_chunk.strip(),
                            "source_file": file_name,
                            "source_type": "PDF",
                            "page_or_row": f"{page_num}페이지"
                        })
                    # 중첩 영역 설정
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + "\n\n" + para
                else:
                    current_chunk += ("\n\n" if current_chunk else "") + para
                    
        # 잔여 텍스트가 있을 경우 추가
        if current_chunk.strip():
            chunks_with_metadata.append({
                "content": current_chunk.strip(),
                "source_file": file_name,
                "source_type": "PDF",
                "page_or_row": f"{page_num}페이지"
            })
            
    print(f"[성공] PDF 분석 완료: 총 {len(chunks_with_metadata)} 개의 고품질 텍스트 청크 생성 완료.")
    return chunks_with_metadata

def extract_and_chunk_excel(excel_path):
    """
    엑셀 파일을 열어 각 행의 데이터를 칼럼 키-값 형태의 문자열로 변환하고
    해당 행 번호와 파일명을 메타데이터로 매핑하여 추출합니다.
    """
    print(f"[엑셀] '{excel_path}' 읽기 시작...")
    
    file_name = os.path.basename(excel_path)
    df = pd.read_excel(excel_path, engine='openpyxl')
    df = df.fillna("")
    
    chunks_with_metadata = []
    for index, row in df.iterrows():
        row_text_list = []
        for col in df.columns:
            val = str(row[col]).strip()
            # 빈 셀 정보는 제외하여 노이즈 축소 및 토큰 절약
            if val:
                row_text_list.append(f"{col}: {val}")
        
        if not row_text_list:
            continue
            
        chunk_text = ", ".join(row_text_list)
        full_chunk = f"[행 데이터] {chunk_text}"
        
        chunks_with_metadata.append({
            "content": full_chunk,
            "source_file": file_name,
            "source_type": "Excel",
            "page_or_row": f"{index + 1}번째 행"
        })
        
    print(f"[성공] 엑셀 분석 완료: 총 {len(chunks_with_metadata)} 개의 데이터 행(청크) 추출 성공.")
    return chunks_with_metadata

# ==========================================
# 2. Ollama 기반 임베딩 추출 함수
# ==========================================
def get_ollama_embedding(text):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response['embedding']

# ==========================================
# 3. PostgreSQL(pgvector) 초기화 및 데이터 저장
# ==========================================
def init_and_save_to_pgvector(chunks_with_metadata):
    """
    데이터베이스 테이블을 생성하고 임베딩 벡터와 풍부한 메타데이터를 저장합니다.
    최신 pgvector 환경에 부합하도록 탐색 효율이 우수한 HNSW 인덱스를 구축합니다.
    """
    conn = get_db_connection(DB_CONN_STR)
    cur = conn.cursor()
    
    # 1. pgvector 확장 활성화
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    
    # 2. 동적 임베딩 크기 파악
    dim = get_embedding_dimension(EMBED_MODEL)
    
    # 3. 테이블 정의 및 생성
    cur.execute("DROP TABLE IF EXISTS esg_documents;")
    cur.execute(f"""
        CREATE TABLE esg_documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector({dim}),
            source_file TEXT,
            source_type TEXT,
            page_or_row TEXT
        );
    """)
    conn.commit()
    
    print("[시작] Ollama 임베딩 연산 및 pgvector 저장 중 (잠시 시간이 소요됩니다)...")
    data_to_insert = []
    for item in chunks_with_metadata:
        vector = get_ollama_embedding(item["content"])
        data_to_insert.append((
            item["content"], 
            str(vector), 
            item["source_file"], 
            item["source_type"], 
            item["page_or_row"]
        ))
        
    # 복수 행 동시 저장
    execute_values(
        cur, 
        "INSERT INTO esg_documents (content, embedding, source_file, source_type, page_or_row) VALUES %s", 
        data_to_insert
    )
    conn.commit()
    
    # 4. HNSW(Hierarchical Navigable Small World) 고속 인덱스 생성
    print("[인덱스] HNSW 벡터 인덱스 생성 중...")
    cur.execute("CREATE INDEX IF NOT EXISTS esg_documents_hnsw_idx ON esg_documents USING hnsw (embedding vector_cosine_ops);")
    conn.commit()
    
    cur.close()
    conn.close()
    print("[성공] pgvector에 ESG 지식 데이터베이스 및 메타데이터 구축 완료!")

# ==========================================
# 4. pgvector 기반 코사인 유사도 검색 알고리즘
# ==========================================
def search_similar_documents(query, top_k=3):
    """
    질문에 대한 벡터 유사도 탐색을 진행하여 연관된 텍스트와 출처 메타데이터를 함께 가져옵니다.
    """
    query_vector = get_ollama_embedding(query)
    
    conn = get_db_connection(DB_CONN_STR)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT content, source_file, source_type, page_or_row 
        FROM esg_documents 
        ORDER BY embedding <=> %s 
        LIMIT %s;
    """, (str(query_vector), top_k))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    retrieved = []
    for row in results:
        retrieved.append({
            "content": row[0],
            "source_file": row[1],
            "source_type": row[2],
            "page_or_row": row[3]
        })
    return retrieved

# ==========================================
# 5. RAG 생성 파트 (출처 명시)
# ==========================================
def ask_esg_chatbot(model_name, query):
    """
    검색 결과의 메타데이터를 활용하여 출처가 철저하게 명시된 맞춤형 RAG 프롬프트를 조립하고
    Ollama LLM을 통해 답변과 출처 리스트를 반환합니다.
    """
    # 생성용 LLM 모델 유효성 및 자동 다운로드 확인
    check_and_pull_ollama_model(model_name)
    
    retrieved_contexts = search_similar_documents(query, top_k=3)
    
    if not retrieved_contexts:
        return "[경고] 검색 결과가 존재하지 않아 답변을 구성할 수 없습니다.", []
    
    # 텍스트 및 메타데이터 포맷팅
    formatted_context_list = []
    citations = []
    
    for idx, ctx in enumerate(retrieved_contexts):
        source_info = f"{ctx['source_file']} ({ctx['page_or_row']})"
        citations.append(source_info)
        
        formatted_context_list.append(
            f"[참고 자료 {idx+1}] (출처: {source_info})\n내용: {ctx['content']}"
        )
        
    context = "\n\n".join(formatted_context_list)
    
    # 출처를 본문에 명시하도록 하는 시스템 지침 결합
    prompt = f"""당신은 ESG 공급망 실사 지침 및 글로벌 규제 전문가입니다. 
주어진 [참고 문서]의 핵심 내용을 기반으로 [사용자 질문]에 신뢰할 수 있는 정확한 정보를 제공하세요.
반드시 제공된 문서에 나와 있는 내용만을 토대로 요약 및 분석을 수행해야 합니다.
답변할 때 해당하는 내용이 어떤 참고 자료(예: [참고 자료 1], [참고 자료 2] 등)에서 온 것인지 본문 내에 명시하여 사용자가 출처를 검증할 수 있도록 하세요.

[참고 문서]
{context}

[사용자 질문]
{query}
"""
    
    print(f"\n[AI] [{model_name}] 모델이 답변을 생성하는 중...")
    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        return response['response'], citations
    except Exception as e:
        err_msg = f"LLM 답변 생성 중 치명적인 실패가 발생했습니다: {e}"
        return err_msg, citations

# ==========================================
# 6. pgvector 백업 및 허깅페이스 저장
# ==========================================
def export_pgvector_to_file_and_hf(repo_id="Makesols/esg-vector-dataset"):
    """
    DB 내용을 로컬 파일(CSV, Parquet)로 백업하고 필요 시 Hugging Face Hub에 업로드합니다.
    """
    conn = get_db_connection(DB_CONN_STR)
    cur = conn.cursor()
    
    cur.execute("SELECT id, content, embedding::text, source_file, source_type, page_or_row FROM esg_documents;")
    rows = cur.fetchall()
    
    cur.close()
    conn.close()
    
    if not rows:
        print("[경고] DB에 저장된 데이터가 없어 백업을 건너뜁니다.")
        return

    df = pd.DataFrame(rows, columns=['id', 'content', 'embedding', 'source_file', 'source_type', 'page_or_row'])
    # 스트링화된 벡터를 실수형 리스트로 원복
    df['embedding'] = df['embedding'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])

    df.to_csv("esg_vector_backup.csv", index=False, encoding='utf-8-sig')
    df.to_parquet("esg_vector_backup.parquet", index=False)
    print("[백업] 로컬 백업 완료: 'esg_vector_backup.csv' 및 '.parquet' 파일이 성공적으로 보존되었습니다.")

    if repo_id:
        print("[HF] 허깅페이스 데이터셋 변환 및 업로드 시작...")
        hf_dataset = Dataset.from_pandas(df)
        hf_dataset.push_to_hub(repo_id, private=True)
        print(f"[성공] 허깅페이스 저장소({repo_id})에 성공적으로 업로드되었습니다!")


# ==========================================
# 메인 제어 흐름
# ==========================================
if __name__ == "__main__":
    all_chunks = []

    # 1. PDF 폴더 처리
    pdf_folder_path = "./esg_pdf_files"
    if not os.path.exists(pdf_folder_path):
        os.makedirs(pdf_folder_path)
        print(f"[폴더] '{pdf_folder_path}' 폴더가 생성되었습니다. 분석할 PDF 파일들을 이 폴더에 넣어주세요.")
    
    pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    if pdf_files:
        print(f"[문서] {len(pdf_files)}개의 PDF 파일을 찾았습니다. 분석을 시작합니다.")
        for pdf_path in pdf_files:
            try:
                pdf_chunks = extract_and_chunk_pdf(pdf_path)
                all_chunks.extend(pdf_chunks)
            except Exception as e:
                print(f"[오류] {pdf_path} 파일 읽기 실패: {e}")
    else:
        print(f"[안내] '{pdf_folder_path}' 폴더에 PDF 파일이 없습니다.")

    # 2. 엑셀 폴더 처리
    excel_folder_path = "./esg_excel_files"
    if not os.path.exists(excel_folder_path):
        os.makedirs(excel_folder_path)
        print(f"[폴더] '{excel_folder_path}' 폴더가 생성되었습니다. 분석할 엑셀 파일들을 이 폴더에 넣어주세요.")
        
    excel_files = glob.glob(os.path.join(excel_folder_path, "*.xlsx")) + glob.glob(os.path.join(excel_folder_path, "*.xls"))
    if excel_files:
        print(f"[엑셀] {len(excel_files)}개의 엑셀 파일을 찾았습니다. 분석을 시작합니다.")
        for excel_path in excel_files:
            try:
                excel_chunks = extract_and_chunk_excel(excel_path)
                all_chunks.extend(excel_chunks)
            except Exception as e:
                print(f"[오류] {excel_path} 파일 읽기 실패: {e}")
    else:
        print(f"[안내] '{excel_folder_path}' 폴더에 엑셀 파일이 없습니다.")

    # 3. pgvector 통합 저장
    if all_chunks:
        try:
            init_and_save_to_pgvector(all_chunks)
        except Exception as e:
            print(f"[오류] DB 연동 실패로 지식 구축 건너뜀: {e}")
    else:
        print("[경고] 폴더들에 읽어올 파일이 하나도 있어 DB 구축을 건너뜁니다.")

    # 4. RAG 질의응답 및 백업 수행
    if all_chunks:
        test_query = "협력사의 탄소 배출량 실사 의무 규정과 제재 조치는 어떻게 되나요?"
        
        # 표준 최신 배포 모델 gemma4:e2b 사용 (환경에 따라 자동 검증)
        target_model = "gemma4:e2b" 
        try:
            gemma4_answer, citations = ask_esg_chatbot(target_model, test_query)
            print(f"\n========= [답변] {target_model} 답변 =========")
            print(gemma4_answer)
            print("\n========= [문서] 답변 생성에 참고한 출처 리스트 =========")
            for idx, citation in enumerate(citations):
                print(f"[{idx+1}] {citation}")
        except Exception as e:
            print(f"\n[오류] LLM 답변 생성 중 오류 발생: {e}")
            print("[팁] Ollama 서비스가 실행 중이고 해당 LLM 모델이 존재하는지 체크해보세요.")

        try:
            export_pgvector_to_file_and_hf(repo_id="Makesols/esg-vector-dataset")
        except Exception as e:
            print(f"[오류] 백업/허깅페이스 내보내기 실패: {e}")