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
  - **임베딩 생성·저장·조회**: 기존 구축된 **AIPro+** API를 호출하여 처리 (직접 구현하지 않음 — 아래 2.1절 참고)
  - **분류 학습**: Scikit-learn Logistic Regression 모델 학습 및 추론

### 2.1 임베딩 인프라 — 기존 AIPro+ 활용

임베딩 모델 구동이나 벡터 저장소 관리를 직접 구현하지 않고, 이미 운영 중인 **AIPro+**(RAG 서비스, `localhost:28000`)의 API를 호출하여 처리한다. 이 서비스는 내부적으로 **BGE-M3** 임베딩 모델과 **Qdrant** 벡터 DB를 사용한다.

| 용도 | API 태그 | 엔드포인트 | 설명 |
|---|---|---|---|
| 도메인(분류) 관리 | RAG Management | `POST /api/domains` | 데이터를 그룹화할 도메인 생성. **프로젝트당 1개 고정 도메인**(`DOMAIN_NAME` 상수, 예: `embedding_lr`)만 사용하며, 최초 실행 시 1회 생성(이미 존재하면 무시)하고 이후 모든 콜렉션이 이 도메인 하위에 귀속됨 |
| 콜렉션 생성·관리 | RAG Management | `POST /api/collections` | 벡터 공간(콜렉션) 생성. 위 고정 도메인 하위에, 입력 데이터 경로(`data/<version>/{train,test,val}.jsonl`)에서 자동 추출한 **데이터 version** + **용도(train/test/validation) 구분**을 조합한 `<version>_<train\|test\|validation>` 값을 `name`으로 사용 — 데이터 버전 × 용도별로 별도 콜렉션으로 분리됨. 실행 시마다 존재 여부를 먼저 확인하고, 이미 존재하면 재생성하지 않고 그대로 사용(idempotent) |
| 임베딩 벡터 생성 | LLM | `POST /api/embeddings` | 텍스트 목록 → 임베딩 벡터 배열 반환 |
| 지식 데이터 등록 | RAG Data | `POST /api/rag/knowledge` | 임베딩 + 메타데이터를 벡터 저장소에 적재 (Upsert 지원) |
| 유사도 검색 | RAG Data | `POST /api/rag/search` | 쿼리 벡터와 유사한 데이터 검색 (코사인 유사도) |
| 일괄 삭제 | RAG Data | `DELETE /api/rag/bulk-delete` | 조건부 대량 삭제 (재학습 시 데이터 정화용) |

- **인증**: Bearer Token (HTTPBearer) 방식
- **핵심 이점**: 임베딩 모델 로딩, GPU/메모리 관리, 벡터 인덱싱을 모두 기존 인프라에 위임하므로 본 프로젝트는 **분류 로직에만 집중** 가능
- **사전 등록 순서**: 임베딩 요청 전에 반드시 ①도메인(`DOMAIN_NAME`, 최초 1회) → ②콜렉션(`<version>_<split>`, 버전/용도마다) 순으로 존재를 보장한 뒤에만 `POST /api/rag/knowledge` 적재를 수행한다 — 콜렉션·도메인이 없는 상태로 지식 데이터를 등록할 수 없다. 둘 다 **이미 존재하면 재등록(재생성)하지 않고 기존 것을 그대로 사용**한다 — 매 실행마다 존재 확인 후 없을 때만 생성.
- **지식 데이터 등록 전략(레코드 단위 중복 판별 없음)**:
  - Train/Test/Validation 각 데이터셋 전체를 한 번의 배치로 콜렉션에 통째로 적재하는 방식이므로, 해시 비교 등 레코드 단위 중복 판별은 두지 않는다 — 대신 콜렉션 자체를 데이터 version × 용도(train/test/validation) 단위로 분리해 관리한다(위 콜렉션 표 참고).
  - 지식 데이터 등록(`POST /api/rag/knowledge`) 시 `source` 필드에는 **분류 라벨값**(`카테고리`, 5-class 중 하나: `IT`/`DAILY`/`KNOWLEDGE`/`CREATIVE`/`ANOMALY`)을 저장한다 — 이를 통해 콜렉션 내에서도 라벨 기준으로 데이터를 조회·추적할 수 있다.
  - 재학습(하이퍼파라미터 조정 등) 시에는 동일 콜렉션(`<version>_<split>`)을 대상으로 전체 재임베딩·재적재(Upsert)를 수행한다.

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
    [AIPro+] POST /api/embeddings
         │
    1024D 벡터 배열
         │
    [적재] source=label(카테고리) 로 Qdrant 적재 (collection=<version>_<train|test|validation>)
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
  [AIPro+] POST /api/embeddings → 1024D 벡터
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
| 임베딩 변환 및 적재 | AIPro+ API 호출로 1024D 벡터 생성, source=라벨값 저장, train/test/validation별 콜렉션 분리 적재 |
| 분류 모델 학습 | Scikit-learn Logistic Regression, 하이퍼파라미터 탐색, 테스트셋 기반 튜닝 |
| 성능 검증 | 검증셋 200건 대상 Accuracy/F1 평가, Confusion Matrix, 목표 미달 시 루프백 |
| 추론 파이프라인 | 텍스트 입력 → 전처리 → 임베딩 → LR 분류 → 카테고리 + 신뢰도 출력 |
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
