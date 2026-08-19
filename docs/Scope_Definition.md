# embedding을 lr로 분류하기 - Scope Definition (v1)

## 1. 프로젝트 개요
- **목적**: 기 구축된 임베딩 모델과 가벼운 머신러닝 알고리즘(Logistic Regression)을 결합하여 실시간 쿼리 분류 파이프라인(분류 모델)을 구축함.

## 2. 분류 시스템 핵심 아키텍처 및 로직
- **분류 카테고리 정의** (5-class multi-class):

  NON_IT는 "IT가 아닌 모든 것"이라 패턴이 방대하므로, 4개 하위 카테고리로 세분화하여 각각 독립된 클러스터로 학습시킨다. 최종 판정 시에는 IT 외 4개를 NON_IT로 집계한다.

  | 라벨 | 설명 | 최종 판정 |
  |---|---|---|
  | `IT` | 5개 IT 직무 역할 기반 기술 질의 (미들웨어/OS/네트워크/DBA/DevOps) | **IT** |
  | `DAILY` | 일상 대화 — 날씨, 맛집, 여행, 인사, 감정 대화 등 | NON_IT |
  | `KNOWLEDGE` | 일반 지식/교양 — 과학, 역사, 수학, 시사, 경제 상식 | NON_IT |
  | `CREATIVE` | 창작/엔터테인먼트 — 영화, 음악, 게임, 스포츠, 소설 | NON_IT |
  | `ANOMALY` | 무의미 입력 — 랜덤 문자열, 의미 없는 텍스트, 반복 문자 | NON_IT |
- **기술 스택 및 프로세스**:
  - **텍스트 전처리**: 임베딩에 불필요하거나 분류 판단을 왜곡시키는 **형식적 노이즈만** 제거하고, 의미 있는 본문 신호는 보존한다 — ①마크다운 코드펜스 구분자(\`\`\`, \`\`\`bash 등)만 제거하고 본문 명령어는 유지(백틱 유무가 IT를 암시하는 표면적 지름길 학습 방지, [[P1_Data_Preprocessing_Review]] 3.2절 참고) ②스택 트레이스/트레이스백 라인(`Exception`, `Caused by`, `at ...(`, `Traceback`) 제거 ③과도한 공백·연속 개행 정규화(collapse)
  - **임베딩 생성·저장·조회**: 기존 구축된 **AIPro+**와 별개의 **Embedding Service**를 호출하여 처리 (직접 구현하지 않음 — 아래 2.1절 참고)
  - **분류 학습**: Scikit-learn Logistic Regression 모델 학습 및 추론

### 2.1 임베딩 인프라 — AIPro+(학습 전용) + 독립 Embedding Service(추론 전용)

이 프로젝트는 서로 무관한 **두 개의 외부 서비스**를 쓴다. 혼동하기 쉬우므로 명확히
구분한다:

- **AIPro+**(RAG 서비스, `localhost:28000`) — **Phase 2(학습)만** 사용. 지식 데이터를
  콜렉션에 텍스트(`content`)로 등록하면 AIPro+가 내부적으로 **BGE-M3** 임베딩을 계산해
  **Qdrant**에 저장하고, 등록된 데이터를 임베딩 포함해서 일괄 조회할 수 있다. 이
  프로젝트 코드는 AIPro+가 별도로 제공하는 `POST /api/embeddings`(임베딩만 단독 계산)는
  **호출하지 않는다** — 아래 API 표의 "사용 여부" 열 참고.
- **Embedding Service**(독립 서비스, `localhost:8000`, AIPro+와 무관) — **Phase 5(추론)만**
  사용. 실시간 쿼리 1건이 들어올 때마다 Qdrant 저장 없이 임베딩 벡터만 즉시 얻기 위해
  직접 호출한다. 인증이 필요 없다.

두 서비스를 나눈 이유: Phase 2는 데이터 전체를 한 번에 등록·조회하는 배치 작업이라
AIPro+의 지식 저장소 기능(도메인/콜렉션/추적성)이 그대로 유용하지만, Phase 5는 쿼리
1건마다 저장할 필요가 없는 실시간 요청이라 AIPro+를 거치지 않고 임베딩 계산만 하는
가벼운 서비스를 직접 호출하는 것이 더 적합하다.

#### AIPro+ API (`localhost:28000`) — Phase 2 전용

| 용도 | API 태그 | 엔드포인트 | 사용 여부 | 설명 |
|---|---|---|---|---|
| 도메인(분류) 관리 | RAG Management | `POST /api/domains` | 사용 | 데이터를 그룹화할 도메인 생성. **프로젝트당 1개 고정 도메인**(`DOMAIN_NAME` 상수, 예: `embedding_lr`)만 사용하며, 최초 실행 시 1회 생성(이미 존재하면 무시)하고 이후 모든 콜렉션이 이 도메인 하위에 귀속됨 |
| 콜렉션 생성·관리 | RAG Management | `POST /api/collections` | 사용 | 벡터 공간(콜렉션) 생성. 위 고정 도메인 하위에, 입력 데이터 경로(`data/<version>/{train,test,val}.jsonl`)에서 자동 추출한 **데이터 version** + **용도(train/test/validation) 구분**을 조합한 `<version>_<train\|test\|validation>` 값을 `collection_name`으로 사용 — 데이터 버전 × 용도별로 별도 콜렉션으로 분리됨. 실행 시마다 존재 여부를 먼저 확인하고, 이미 존재하면 재생성하지 않고 그대로 사용(idempotent). **주의**: `collection_name`은 AIPro+가 `^[a-zA-Z0-9_-]+$` 패턴만 허용(점 `.` 불가 — 실제로 `POST /api/collections`에 점이 든 이름을 보내면 422 검증). 데이터 버전 문자열은 `v0.1`/`v0.2`처럼 점을 포함하므로, `collection.py`가 콜렉션명을 만들 때 버전 문자열의 점(`.`)을 언더스코어(`_`)로 치환한다 — 예: `v0.2` + `train` → `v0_2_train`(사용자 확인, 2026-08-19, 실제 AIPro+ 호출로 422 재현 후 결정) |
| 임베딩 벡터 생성 | LLM | `POST /api/embeddings` | **미사용** | AIPro+ 자체 임베딩 단독 계산 API. 이 프로젝트는 학습 경로에서도 지식 데이터 등록(`content` 기반, 아래)과 조회만으로 벡터를 얻으므로 이 API를 직접 호출하지 않는다 |
| 지식 데이터 등록 | RAG Data | `POST /api/rag/knowledge` | 사용 | `content`(텍스트) + 메타데이터를 등록하면 AIPro+가 내부에서 임베딩을 계산해 벡터 저장소에 적재(Upsert 지원) — 등록 요청 자체는 벡터를 받지 않는다 |
| 지식 데이터 조회(임베딩 포함) | RAG Data | `GET /api/rag/knowledge` | 사용 | `domain_id`(필수) + `collection`/`source`(선택, 부분일치) 조건으로 등록된 데이터를 임베딩 벡터 포함하여 조회(`limit` 기본 50). `POST /api/rag/search`는 임베딩 값을 반환하지 않아, 등록된 벡터를 다시 얻으려면 이 API를 쓴다 |
| 유사도 검색 | RAG Data | `POST /api/rag/search` | 미사용(참고용) | 쿼리 벡터와 유사한 데이터 검색 (코사인 유사도). 응답에 임베딩 벡터는 포함되지 않음 |
| 일괄 삭제 | RAG Data | `DELETE /api/rag/bulk-delete` | 미사용(참고용) | 조건부 대량 삭제 (재학습 시 데이터 정화용) |

- **인증**: Bearer Token (HTTPBearer) 방식
- **사전 등록 순서**: 지식 데이터 등록 전에 반드시 ①도메인(`DOMAIN_NAME`, 최초 1회) → ②콜렉션(`<version>_<split>`, 버전/용도마다) 순으로 존재를 보장한 뒤에만 `POST /api/rag/knowledge` 적재를 수행한다 — 콜렉션·도메인이 없는 상태로 지식 데이터를 등록할 수 없다. 둘 다 **이미 존재하면 재등록(재생성)하지 않고 기존 것을 그대로 사용**한다 — 매 실행마다 존재 확인 후 없을 때만 생성.
- **지식 데이터 등록·조회 전략(콜렉션 단위 판별, 레코드 단위 중복 판별 없음)**:
  - Phase 2 파이프라인은 임베딩을 직접 계산하지 않는다 — `text_cleaner`로 정제한 텍스트를 그대로 `POST /api/rag/knowledge`(`content`, `source`=분류 라벨값)로 등록하면 AIPro+가 내부에서 임베딩을 계산·저장하고, 그 뒤 `GET /api/rag/knowledge`로 해당 콜렉션 전체를 일괄 조회해 `embedding`+`source`를 그대로 `*_vectors.parquet`으로 저장한다.
  - Train/Test/Validation 각 데이터셋 전체를 한 번의 배치로 콜렉션에 통째로 등록하는 방식이므로, 해시 비교 등 **레코드 단위** 중복 판별은 두지 않는다 — 대신 콜렉션 자체를 데이터 version × 용도(train/test/validation) 단위로 분리해 관리한다(위 콜렉션 표 참고).
  - `source` 필드에는 **분류 라벨값**(`카테고리`, 5-class 중 하나: `IT`/`DAILY`/`KNOWLEDGE`/`CREATIVE`/`ANOMALY`)을 저장한다 — 이를 통해 콜렉션 내에서도 라벨 기준으로 데이터를 조회·추적할 수 있다.
  - **재실행 시 콜렉션 단위 재등록 스킵**: Phase 2 재실행 시, 등록을 수행하기 전에 `GET /api/rag/knowledge`(`domain_id`+`collection`, `limit`=해당 split의 입력 레코드 수 이상)로 해당 콜렉션에 이미 등록된 건수를 조회한다. 조회된 건수가 입력 JSONL의 레코드 수와 **일치하면** `POST /api/rag/knowledge` 재등록을 건너뛰고 방금 조회한 결과(`embedding`+`source`)를 그대로 `*_vectors.parquet` 생성에 쓴다. 건수가 다르거나(0건 포함) 콜렉션이 비어 있으면 **콜렉션 전체**를 재등록(Upsert)한 뒤 다시 `GET`으로 조회한다 — 레코드 단위 비교(해시 등)는 하지 않고 콜렉션 단위 건수 비교로만 판단한다.
  - 하이퍼파라미터 조정 등으로 **동일 데이터에 대해 임베딩만 다시 계산**해야 하는 경우(임베딩 모델이 바뀌는 등)는 위 스킵 조건에 해당하지 않도록 콜렉션명(`<version>_<split>`)에 버전을 새로 부여해 별도 콜렉션으로 재등록해야 한다 — 같은 콜렉션명으로는 강제 재등록 수단을 별도로 제공하지 않는다.

#### Embedding Service API (`localhost:8000`, AIPro+와 별개) — Phase 5 전용

| 용도 | 엔드포인트 | 설명 |
|---|---|---|
| 헬스체크 | `GET /health` | 서비스 상태 확인 |
| 임베딩 생성 | `POST /embed` | `{"texts": [...]}` → `{"embeddings": [[...]], "dim": 1024, "count": N}`. 인증 불필요. Phase 5 추론 경로(`inference/predictor.py`)가 쿼리 1건이 들어올 때마다 직접 호출해 벡터만 얻고, Qdrant 등록은 하지 않는다 |

## 3. 학습 데이터 확보 전략

실 운영 로그를 사용할 수 없는 환경이므로, **LLM을 활용한 합성 데이터 생성** 방식으로 학습 데이터를 확보한다. 이 절이 설명하는 생성 방식은 프롬프트를 가지고 LLM과 상호작용하며 콘텐츠를 만드는 과정으로, **이 저장소의 Phase 1 코드가 수행하는 작업이 아니다** — 그렇게 만들어진 결과물(현재는 CSV, `data/<version>/role_*.csv`)이 "이미 확보된 원본"이고, Phase 1 코드는 그 원본을 학습 파이프라인이 쓰는 형식(JSONL)으로 변환·정리하는 것부터 시작한다([[P1_설계서_DataPreparation]] 참고).

### 3.1 IT 데이터 — 역할 기반 프롬프트 생성

실무 현장에서 실제로 발생하는 질의-응답 패턴을 재현하기 위해 **5개 IT 직무 역할**을 정의하고, 각 역할별 프롬프트를 LLM에 입력하여 데이터를 생성한다.

| # | 역할 | 주요 기술 스택 | 질의 예시 |
|---|---|---|---|
| 1 | Middleware / Application 운영 | JEUS 8, Tomcat, WebtoB, nginx, Apache | GC 튜닝, 스레드 덤프 분석, 502/504 에러, 커넥션 릭 |
| 2 | OS / 서버 인프라 | AIX, RHEL, Windows Server, LVM, JFS2 | vmstat 해석, LVM 확장, ulimit, 커널 파라미터 |
| 3 | Network / Security | L2/L3 스위치, 방화벽, SSL/TLS, DNS | tcpdump, 인증서 체인, LB 헬스체크, 세션 타임아웃 |
| 4 | DBA | Oracle, Tibero, PostgreSQL, MySQL | 슬로우 쿼리, 락 경합, 테이블스페이스 부족, 실행계획 |
| 5 | Cloud / Container / DevOps | Docker, K8s, Jenkins, Helm, Prometheus | Pod CrashLoop, HPA, PV 마운트, 파이프라인 실패 |

- 프롬프트 파일은 `prompt/` 디렉터리에 IT 공통 설정(`common_it.md`)과 역할별 파일(`roles/01~05`)로 분리·관리
- 난이도 분포: 단순 커맨드/개념 50% + 실무 트러블슈팅 50%
- 응답 스타일: 반말, 간결체, 코드블록 포함, 본론만

### 3.2 NON IT 데이터 — 4개 하위 카테고리별 프롬프트 생성

NON_IT를 4개 하위 카테고리로 세분화하여 각각 별도의 프롬프트로 데이터를 생성한다.

| 라벨 | 생성 방향 | 질의 예시 |
|---|---|---|
| `DAILY` | 일상 대화, 감정 표현, 인사, 날씨/맛집/여행 | "오늘 점심 뭐 먹지?", "요즘 너무 피곤해" |
| `KNOWLEDGE` | 일반 교양, 과학/역사/수학, 시사 | "광합성 원리 알려줘", "조선시대 왕 순서" |
| `CREATIVE` | 영화/음악/게임 추천, 스포츠, 소설 창작 | "올해 볼만한 영화 추천", "단편소설 써줘" |
| `ANOMALY` | 무의미 문자열, 반복 문자, 랜덤 입력 | "asdfkjh 123", "ㅋㅋㅋㅋㅋㅋ", "aaabbb" |

- 프롬프트 파일은 `prompt/` 디렉터리에 NON_IT 공통 설정(`common_non_it.md`)과 역할별 파일(`roles/06~09`)로 관리

### 3.3 데이터 규모 및 분할

5개 클래스 × 200건 = 총 1,000건. 클래스 간 **균등 분배**로 구성한다.

| 구분 | 클래스당 | × 5 클래스 | 합계 | 용도 |
|---|---|---|---|---|
| 학습(Train) | 120건 | 600건 | 600 | 모델 학습 |
| 테스트(Test) | 40건 | 200건 | 200 | 하이퍼파라미터 튜닝 |
| 검증(Validation) | 40건 | 200건 | 200 | 최종 정확도 평가 |

- 출력 형식: **JSONL**(레코드 1개 = JSON 객체 1줄, 키: `질의`, `응답`, `카테고리`) — 애초 CSV로
  계획했으나, 멀티라인·따옴표·쉼표가 섞인 응답 텍스트에서 CSV 이스케이프 규칙(RFC4180)이
  실제로 데이터 손상을 일으킨 사고([[P1_Data_Preprocessing_Review]] 3.1절)가 있어 이런
  이스케이프 처리 자체가 필요 없는 JSONL로 전환한다
- 카테고리 값: `IT`, `DAILY`, `KNOWLEDGE`, `CREATIVE`, `ANOMALY`
- 최종 리포팅 시 IT 외 4개 카테고리를 NON_IT로 집계하여 이진 성능도 함께 측정

## 4. 분류 모델 설계

### 4.1 모델 아키텍처

- **입력**: BGE-M3 임베딩 벡터 (1024 차원, `float32`)
- **모델**: Scikit-learn `LogisticRegression` (multi-class, `multi_class='multinomial'`)
- **출력**: 5-class 분류 (`IT`, `DAILY`, `KNOWLEDGE`, `CREATIVE`, `ANOMALY`) + 각 클래스별 확률값 (`predict_proba()`)
- **실행 환경**: CPU 전용 (GPU 불필요)
- **분류 전략**:
  - 5개 클래스를 균등하게 학습하여 각 카테고리의 결정 경계를 정밀하게 형성
  - NON_IT를 4개 하위 카테고리로 세분화함으로써, 임베딩 공간에서 각각 독립된 클러스터로 학습
  - 최종 판정 시 `predict_proba()` 최대 확률 클래스가 IT이면 IT, 나머지 4개 중 하나이면 NON_IT로 집계
  - BGE-M3의 1024차원 임베딩이 의미적 분리를 담당하고, LR은 그 위에 선형 경계만 학습

### 4.2 학습 파이프라인

```
JSONL 데이터 (1,000건, 5 클래스 × 200건)
  │
  ├─ 학습셋 600건 (클래스당 120건) ──┐
  ├─ 테스트셋 200건 (클래스당 40건) ─┤
  └─ 검증셋 200건 (클래스당 40건) ──┘
         │
    [사전 등록] 도메인(DOMAIN_NAME, 최초 1회) → 콜렉션(<version>_<train|test|validation>)
         │
    [AIPro+] POST /api/rag/knowledge — content=정제 텍스트, source=label(카테고리)
         │   (AIPro+가 내부에서 임베딩 계산 후 Qdrant 적재, collection=<version>_<train|test|validation>)
    [AIPro+] GET /api/rag/knowledge — 콜렉션 일괄 조회
         │
    1024D 벡터 배열(embedding) + label(source)
         │
    Multi-class Logistic Regression 학습 (5 클래스)
         │
    테스트셋으로 하이퍼파라미터 튜닝
         │
    검증셋으로 최종 성능 평가 (5-class + IT vs NON_IT 집계)
         │
    모델 저장 (.pkl)
```

### 4.3 하이퍼파라미터 탐색

| 파라미터 | 탐색 범위 | 기본값 |
|---|---|---|
| `C` (정규화 강도) | 0.01, 0.1, 1.0, 10.0 | 1.0 |
| `solver` | `lbfgs`, `liblinear` | `lbfgs` |
| `max_iter` | 500 ~ 2000 | 1000 |

- 테스트셋 200건에 대한 정확도(Accuracy)와 F1-Score 기준으로 최적 조합 선택
- 탐색은 `GridSearchCV` 또는 수동 비교 방식 적용

### 4.4 평가 지표

| 지표 | 설명 | 목표 |
|---|---|---|
| 5-class Accuracy | 5개 클래스 전체 정답 비율 | ≥ 85% |
| IT vs NON_IT Accuracy | 4개 NON_IT를 합산한 이진 정답 비율 | ≥ 90% |
| F1-Score (macro) | 5개 클래스 F1의 평균 | ≥ 0.85 |
| Confusion Matrix | 5-class 오분류 패턴 + IT vs NON_IT 집계 | 시각화 출력 |
| Classification Report | 클래스별 precision, recall, f1, support | 로그 출력 |

### 4.5 검증 실패 시 루프백

검증셋 200건 대상 평가에서 목표 미달 시, 원인에 따라 이전 Phase로 회귀한다.

| 원인 | 대응 | 돌아갈 Phase |
|---|---|---|
| 데이터 품질 문제 (라벨 오류, 모호한 경계 데이터) | LLM으로 원본 재생성 또는 재라벨링(코드 외부 과정) 후 Phase 1로 재변환 | Phase 1 (데이터 준비) |
| 모델/하이퍼파라미터 문제 | 기존 임베딩 재사용, LR 파라미터만 재조정 | Phase 3 (모델 학습) |
| 특정 유형의 오탐 집중 | 오탐 사례 분석 후 해당 유형 데이터 보강 | Phase 1 → Phase 3 |

## 5. 추론 파이프라인 설계

학습 완료된 모델을 사용하여 새로운 텍스트를 실시간으로 분류하는 흐름.

```
입력 텍스트 (질의 + 응답)
      │
  텍스트 전처리 (노이즈 제거)
      │
  [Embedding Service, localhost:8000] POST /embed → 1024D 벡터 (AIPro+ 미사용)
      │
  학습된 LR 모델 로드 (.pkl)
      │
  predict_proba() → [IT, DAILY, KNOWLEDGE, CREATIVE, ANOMALY 확률]
      │
  최대 확률 클래스 판정
      │
  IT → IT  /  그 외 → NON_IT (상세 카테고리도 함께 반환)
```

- 모델 파일(`.pkl`)은 `joblib`으로 직렬화하여 저장·로드
- 추론 시 임베딩 API 호출 1회 + LR 예측 1회로 완료 (경량, 밀리초 단위)
- 반환값: 최종 판정(IT/NON_IT) + 상세 카테고리 + 각 클래스별 확률(신뢰도)

## 6. Scope 요약

| 구현 항목 | 상세 |
|---|---|
| 합성 데이터 확보(코드 외부) | LLM 프롬프트 기반, 5 클래스 × 200건 균등 분배 — 이미 완료된 원본(현재 CSV), 이 저장소 코드가 생성하지 않음 |
| 데이터 준비(Phase 1, 코드) | 원본(현재 CSV) → JSONL 변환 → 재조합·클래스별 3:1:1 분할, 1,000건 |
| 임베딩 변환 및 적재 | AIPro+ 지식 데이터 등록(content 기반, source=라벨값)으로 train/test/validation별 콜렉션 분리 적재 → AIPro+가 계산·저장한 1024D 벡터를 일괄 조회(GET)해 확보 |
| 분류 모델 학습 | Scikit-learn Logistic Regression, 하이퍼파라미터 탐색, 테스트셋 기반 튜닝 |
| 성능 검증 | 검증셋 200건 대상 Accuracy/F1 평가, Confusion Matrix, 목표 미달 시 루프백 |
| 추론 파이프라인 | 텍스트 입력 → 전처리 → 독립 Embedding Service(localhost:8000)로 임베딩 → LR 분류 → 카테고리 + 신뢰도 출력 (AIPro+ 미사용) |
| **최종 산출물** | **추론 가능한 파이프라인 코드 + 학습된 모델 파일(`.pkl`)** |

## 7. Phase 진행 로드맵

총 5개의 Phase로 분할 진행되며, 각 Phase는 `main` 브랜치 분기 → PR → 리뷰 → 병합의 형태로 수행됩니다.

| Phase | 명칭 | 작업 내용 | 주요 산출물 | 브랜치(예시) |
|---|---|---|---|---|
| **Phase 0** | **범위 정의 + 공통 모듈 기반 구축** | 작업 Scope 확정 및 본 문서 통합, Phase 1~5 공용 모듈(`config`/`constants`/`domain`/`exceptions`/`workflow`/`logging_config`) 설계·구현 | Scope Definition 문서, `P0_설계서_Common.md`/`P0_설계서_Logging.md`/`P0_테스트결과서_Common.md`, 공통 모듈 코드+테스트 | `feature/phase0` |
| **Phase 1** | **데이터 준비** | 이미 확보된 원본(현재 CSV)을 JSONL로 변환·조합·클래스별 3:1:1 분할 | `data.jsonl`, `train/test/val.jsonl` (1,000건, 라벨 포함) | `feature/phase1` |
| **Phase 2** | **임베딩 변환** | AIPro+ API 호출로 1024D 벡터 변환, source=라벨값 저장, train/test/validation별 콜렉션 분리 적재 | 임베딩 스크립트, 콜렉션/적재 로직 | `feature/phase2` |
| **Phase 3** | **모델 학습** | 1024D 벡터 기반 Logistic Regression 학습, 하이퍼파라미터 탐색 | 학습 스크립트, `.pkl` 모델 파일 | `feature/phase3` |
| **Phase 4** | **검증·정화** | 검증셋 평가 (Accuracy/F1), 오탐 분석, 루프백 정화 | 평가 리포트, 추론 파이프라인 코드 | `feature/phase4` |

## 8. 작업 원칙 (Golden Rules)
1. **요구사항-산출물 페어링**: 채팅으로 생성되는 요구사항(프롬프트)은 `docs/prompts/`에 버저닝하여 저장하고, 이에 대응하는 산출물 역시 쌍(Pair)으로 최신화한다.
2. **이식성(Portability) 확보**: 배포 및 실행 환경 구성 시 패키지 의존성 및 컨테이너 환경을 명확히 명시한다.
3. **추적성 확보**: 데이터 version × 용도(train/test/validation)별로 콜렉션을 분리하고, `source` 필드에 분류 라벨값을 저장하여 재학습·분석 시 콜렉션/라벨 단위로 데이터를 추적할 수 있게 한다.
