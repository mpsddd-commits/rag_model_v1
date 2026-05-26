import os
import sys
import datetime
import ollama
import psycopg2
import numpy as np

# ==========================================
# 0. Windows 콘솔 인코딩 예외 안전장치 (Global Safe Print)
# ==========================================
def safe_print(*args, **kwargs):
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

print = safe_print

# ==========================================
# 1. main.py 모듈 연동 (기존 검색 및 설정 그대로 수입)
# ==========================================
try:
    import main
    from main import (
        search_similar_documents,
        get_db_connection,
        DB_CONN_STR
    )
except ImportError:
    print("[오류] 'main.py' 파일을 찾을 수 없습니다.")
    print("이 검증 스크립트는 기존 코드가 담긴 main.py와 반드시 같은 폴더에 있어야 합니다.")
    sys.exit(1)

# ==========================================
# 2. AI 범위 감지 및 해결책 제시 답변 함수 (기존 포맷 유지)
# ==========================================
def get_integrity_checked_ai_response(model_name, query):
    """
    제공해주신 기존 main.py의 search_similar_documents 함수를 호출하여 
    데이터 정합성을 검증하는 핵심 로직입니다.
    """
    # 1) 이미 구축된 벡터 DB에서 데이터 검색 (기존 코드의 함수 호출)
    retrieved_contexts = search_similar_documents(query, top_k=3)
    
    if not retrieved_contexts:
        context = "⚠️ 현재 데이터베이스 내에 연관된 참고 자료가 존재하지 않습니다."
    else:
        formatted_context_list = []
        for c_idx, ctx in enumerate(retrieved_contexts):
            formatted_context_list.append(
                f"[참고 자료 {c_idx+1}] (출처: {ctx['source_file']} | 위치: {ctx['page_or_row']})\n내용: {ctx['content']}"
            )
        context = "\n\n".join(formatted_context_list)

    # 2) 데이터 정합성 검증 프롬프트 인젝션 (원본 프롬프트 완전 유지)
    strict_prompt = f"""당신은 제공된 문서의 데이터 정합성을 철저히 검증하는 기업용 ESG AI 어시스턴트입니다.

[지침 명령어 - 필수 준수]
1. 입력된 [사용자 질문]이 제공된 [참고 문서]의 지식 범위 안에서 답변이 가능한지 선제적으로 판단하십시오.
2. 만약 질문 내용이 [참고 문서]에 명시되어 있지 않거나, 문서의 핵심 도메인(ESG 공급망 실사, 알루미늄 원자재 전처리/성형, 탄소 배출 규제 등)과 전혀 무관하다면, 절대 임의로 지식을 지어내어 답변하지 마십시오(할루시네이션 금지).
3. 범위에서 벗어난 질문일 경우 반드시 아래 3가지 요소를 포함하여 답변하십시오:
   - [감지 및 거절]: 제공된 참고 문서의 범위를 벗어난 질문임을 명확히 안내.
   - [이유 설명]: 현재 데이터베이스(DB) 컨텍스트에 어떤 데이터가 누락되었거나 부족한지 논리적 원인 분석.
   - [해결책 제시]: 사용자가 원하는 답을 얻기 위해 추후 데이터베이스에 '추가로 적재해야 하는 원본 데이터나 가이드라인의 종류'를 구체적으로 제안.

[참고 문서]
{context}

[사용자 질문]
{query}
"""

    try:
        # 기존 코드 스타일대로 ollama 라이브러리 직접 호출
        response = ollama.generate(model=model_name, prompt=strict_prompt)
        ai_answer = response['response']
        
        # 규칙 기반 정합성 판정 (PASS / PARTIAL / FAIL)
        detection_keywords = ["범위", "포함되어 있지", "제공된 문서", "확인할 수 없", "제한", "알 수 없", "누락"]
        solution_keywords = ["추가", "적재", "확보", "제시", "해결", "필요", "자료", "보완"]
        
        detected = any(kw in ai_answer for kw in detection_keywords)
        solution_provided = any(kw in ai_answer for kw in solution_keywords)
        
        if detected and solution_provided:
            status = "정합성 통과 (PASS - 범위 이탈 감지 및 해결책 제시 완료)"
        elif detected:
            status = "부분 통과 (PARTIAL - 이탈은 감지했으나 구체적 해결책 미흡)"
        else:
            status = "일반 답변 또는 검증 확인 요망 (정상 답변 혹은 할루시네이션 확인 필요)"
            
        return ai_answer, status

    except Exception as e:
        return f"AI 답변 생성 중 오류가 발생했습니다: {e}", "오류 발생 (ERROR)"

# ==========================================
# 3. 질문 입력 인터페이스 및 파일 로그 기록 함수
# ==========================================
def record_and_run_verification(model_name):
    print("\n" + "="*60)
    print(f"🤖 [ESG RAG 정합성 검증 모드] 활성화 (모델: {model_name})")
    print("  • 기존 main.py의 pgvector 데이터 조회를 기반으로 질의를 수행합니다.")
    print("  • 종료하려면 '종료' 또는 'q'를 입력하세요.")
    print("="*60)
    
    log_file = "rag_integrity_test_log.txt"
    
    while True:
        query = input("\n📝 검증할 질문 입력: ").strip()
        if not query:
            continue
        if query.lower() in ['종료', 'q', 'quit', 'exit']:
            print("👋 정합성 검증 시스템을 종료합니다.")
            break
            
        print(f"\n🔍 벡터 DB 대조 및 LLM 연산 중...")
        ai_response, detection_status = get_integrity_checked_ai_response(model_name, query)
        
        # 콘솔 출력 파트
        print("\n" + "-"*50)
        print(f"📊 [정합성 평가]: {detection_status}")
        print("-" * 50)
        print(ai_response)
        print("-" * 50)
        
        # 기록 파트 (질문과 AI 답변, 판정 상태를 텍스트 로그 파일에 저장)
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"=== [검증 일시: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===\n")
                f.write(f"사용자 질문: {query}\n")
                f.write(f"감지 상태: {detection_status}\n")
                f.write(f"AI 답변 및 해결책:\n{ai_response}\n")
                f.write("="*60 + "\n\n")
            print(f"💾 질문 및 답변 결과가 '{log_file}'에 기록되었습니다.")
        except Exception as e:
            print(f"⚠️ 로그 저장 실패: {e}")

# ==========================================
# 메인 단독 실행 제어부
# ==========================================
if __name__ == "__main__":
    # 기존 코드에서 사용 중인 타겟 모델과 동일하게 지정
    TARGET_MODEL = "gemma4:e2b" 
    
    # 즉시 질의 및 검증 로그 생성 로직 가동
    record_and_run_verification(TARGET_MODEL)