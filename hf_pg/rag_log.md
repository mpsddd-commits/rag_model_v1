# RAG 시스템 개발 및 수정 이력 관리 대장 (`rag_log.md`)

이 문서는 `hf_pg` 폴더의 RAG(Retrieval-Augmented Generation) 시스템 고도화 및 향후 모든 코드 수정 사항을 체계적으로 추적하고 기록하는 이력 대장입니다.

---

## 📅 2026-05-25: 1차 RAG 시스템 완벽 고도화 및 안정성 확보

### 1. 수정 목적 및 배경
기존 `main.py`는 PostgreSQL pgvector와 Ollama를 활용한 검색 증강 생성(RAG)의 기본 흐름을 보여주었으나, 데이터베이스 수동 생성 부담, 하드코딩된 벡터 차원으로 인한 이식성 한계, 원본 출처(파일명, 페이지/행 위치) 정보 누락으로 인한 답변 신뢰성 검증의 어려움 등의 문제점이 존재했습니다. 이를 해결하여 상용 등급에 준하는 완벽하고 견고한 RAG 시스템을 구축합니다.

### 2. 주요 개선 항목 (계획)
- **자동 데이터베이스 프로비저닝**: `rag_db`가 없을 경우, 기본 `postgres` DB로 진입하여 자동으로 데이터베이스를 구축하는 자가 치유형 DB 연결 알고리즘 도입.
- **동적 임베딩 차원 감지**: Ollama 모델로부터 테스트 임베딩을 추출하여 실제 차원을 측정하고, 그 결과값으로 pgvector 테이블 컬럼(`embedding vector(dim)`)을 동적 설계.
- **메타데이터 보존형 청크 분할**:
  - **PDF**: 텍스트를 무작위로 자르지 않고 페이지 단위/문장 경계를 최대한 유지하여 청킹하며, 소스 파일명과 페이지 번호를 메타데이터로 함께 수집.
  - **Excel**: 각 행별로 데이터를 파싱하고, 소스 파일명과 행 번호를 매핑하여 수집.
- **pgvector 스키마 확장 & HNSW 인덱싱**:
  - `source_file`, `source_type`, `page_or_row` 메타데이터 필드를 갖추도록 테이블 스키마 개편.
  - 탐색 속도와 효율성이 우수한 `HNSW` 벡터 인덱스 적용.
- **출처 증빙(Citation) 프롬프트 고도화**:
  - 검색된 유사 컨텍스트들에 대해 각각의 원본 파일 및 페이지/행 번호를 구조화된 마크다운 형태로 조립하여 LLM의 RAG 프롬프트에 제공.
  - 최종 챗봇 답변 생성 및 반환 시, 사용자가 직접 눈으로 참고 출처를 비교 대조할 수 있도록 참고 자료 정보를 하단에 가시성 높게 표시.
- **Ollama 환경 예외 처리 보강**:
  - 지정한 임베딩/생성 모델이 없을 시, Ollama에 연동된 모델을 실시간 검사해 없는 모델은 백그라운드에서 `pull` 처리하거나 우회할 수 있는 안정성 제어문 추가.
  - 기존 코드의 오타(`qwen3.5`)를 수정하고 실운용 가능한 배포 모델(`qwen2.5:latest` 또는 다운로드된 대체재)로 구동.

### 3. 세부 구현 함수 현황
1. **`get_db_connection(db_conn_str)`**
   - 데이터베이스 접속을 총괄하며, `rag_db`가 없을 시 `postgres` 기본 데이터베이스에 붙어 `CREATE DATABASE rag_db;` 쿼리를 자동 실행.
   - 데이터베이스 서버 구동 자체가 안 되어 있을 시 유저 친화적인 한글 설명 및 `Docker` 실행 스크립트 출력.
2. **`check_and_pull_ollama_model(model_name)`**
   - 로컬 `ollama` 서비스 내에 `bge-m3` 및 `qwen2.5:latest`가 설치되어 있는지 `list()` 조회를 거치며, 모델 누락 시 자동으로 `ollama.pull()`을 병렬 차단하며 안정적으로 구동하도록 설계.
3. **`get_embedding_dimension(model_name)`**
   - 테스트 문장에 대한 임베딩을 실제 1회 실행하여 차원수(`dim`)를 동적으로 획득. 하드코딩 `1024`로 인한 타 모델 변동 불가 문제 해결.
4. **`extract_and_chunk_pdf(pdf_path, chunk_size, chunk_overlap)`**
   - 페이지 단락(`\n\n`)과 라인 피드를 스캔하여 단락의 맥락이 파괴되지 않게 청킹을 진행.
   - 각 청크에 `source_file`, `source_type` ("PDF"), `page_or_row` ("{N}페이지")를 딕셔너리로 묶어 메타데이터 구축.
5. **`extract_and_chunk_excel(excel_path)`**
   - `openpyxl` 엔진을 통해 한 행씩 파싱하고 비어있는 칼럼 셀은 필터링하여 토큰과 잡음을 최소화한 후, 텍스트 형태(`컬럼명: 값, 컬럼명: 값`)로 가공.
   - 메타데이터에 파일명과 행 번호를 매핑.
6. **`init_and_save_to_pgvector(chunks_with_metadata)`**
   - 수집된 메타데이터를 저장할 수 있도록 `esg_documents` 테이블의 스키마를 고도화 (`source_file TEXT`, `source_type TEXT`, `page_or_row TEXT` 열 추가).
   - 대용량 임베딩의 효율적 색인을 위해 기존 `ivfflat`을 철수하고 정확도와 수렴 속도가 우수한 **`HNSW` 벡터 인덱스** 적용.
7. **`search_similar_documents(query, top_k)`**
   - 쿼리 임베딩 유사도 검색을 거쳐 본문(`content`)뿐 아니라 출처 메타데이터까지 다중 열로 추출하여 딕셔너리 리스트로 반환.
8. **`ask_esg_chatbot(model_name, query)`**
   - 검색된 청크들을 `[참고 자료 N] (출처: 파일명 (상세위치))`의 정교한 마크다운 구조로 결합하여 시스템/RAG 프롬프트 구축.
   - 답변 작성 시 본문 내에서 참고한 대괄호 번호(`[참고 자료 1]`)를 명시하도록 지시하여 모델의 환각 현상(Hallucination) 방지 및 신뢰도 강화.
9. **`export_pgvector_to_file_and_hf(repo_id)`**
   - 스키마 확장에 발맞추어 메타데이터 열까지 포함하여 로컬 CSV/Parquet 백업을 안정적으로 영속화.

### 4. 2차 보완: Windows 콘솔 인코딩 예외 안전장치 (UnicodeEncodeError 방지)
- **발생 현상**: 로컬 Windows(cp949 한국어 시스템)에서 구동할 때, `qwen2.5:latest` 등의 LLM 모델이 생성한 답변 내용에 특수 다국어(Hanja/Chinese 혹은 특정 유니코드 문자)가 포함될 경우, `print` 문을 거치면서 `UnicodeEncodeError: 'cp949' codec can't encode character...` 치명적 크래시가 발생하는 문제 발견.
- **조치 사항**:
  - `main.py` 상단에 `sys` 임포트 및 글로벌 `safe_print` 보조 함수 구현.
  - 내장 `print` 함수를 `print = safe_print`로 전역 덮어쓰기하여, 인코딩이 불가능한 한자/기호가 포함되어 있더라도 시스템 크래시 없이 해당 문자만 `?` 등으로 안전 대체되어 정상 출력이 성공하도록 조치.
- **사용자 수정 반영**:
  - 백업 처리 함수 호출부에 허깅페이스 레포지토리 정보 저장 요청을 반영하여 `export_pgvector_to_file_and_hf(repo_id="Makesols/esg-vector-dataset")`으로 업데이트 완료.

---

올라마 pull: 1.gemma4:e2b 2.qwen3.5:9b