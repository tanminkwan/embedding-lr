# embedding-lr

기 구축된 임베딩 서비스(**AIPro+**, BGE-M3 + Qdrant, `localhost:28000`)와 가벼운
Logistic Regression을 결합해, 실시간 쿼리를 5-class로 분류하는 파이프라인.
자세한 배경은 [docs/Scope_Definition.md](docs/Scope_Definition.md) 참고.

## 분류 대상

| 라벨 | 설명 | 최종 판정 |
|---|---|---|
| `IT` | IT 5개 직무 역할 기반 기술 질의 | **IT** |
| `DAILY` | 일상 대화 | NON_IT |
| `KNOWLEDGE` | 일반 지식/교양 | NON_IT |
| `CREATIVE` | 창작/엔터테인먼트 | NON_IT |
| `ANOMALY` | 무의미 입력 | NON_IT |

## 파이프라인

```
[Phase 1] 데이터 생성        prompt/*.md → role_01~09_*.csv
[Phase 1.5] 데이터 조합/분할  role_*.csv → data.csv → train/test/val.csv
[Phase 2] 임베딩 변환        train/test/val.csv → *_vectors.parquet (AIPro+ 호출)
[Phase 3] 모델 학습          *_vectors.parquet → model_<ver>.pkl (GridSearchCV)
[Phase 4] 검증               val_vectors.parquet + model.pkl → eval_report.md/json
[Phase 5] 추론 서비스        FastAPI 상시 서비스, model_<ver>.pkl 로드 후 실시간 분류
```

전체 구조는 [docs/Architecture_Design.md](docs/Architecture_Design.md) 참고.

## 작업 절차

이 프로젝트는 코드보다 문서가 먼저다. [CLAUDE.md](CLAUDE.md) 3절에 따라 각 Phase(및
공통 모듈)는 아래 순서를 반드시 지킨다.

1. 요구사항정의서 — 무엇을, 왜
2. 설계서 — 입력/출력, 인터페이스, 데이터 흐름
3. 코드 + 테스트 코드 (등급별 TDD 기준은 CLAUDE.md 2절)
4. 테스트 결과서 — 실행 결과, 등급별 커버리지

문서 파일명 규칙: `[P<phase>_]<DocType>_<Topic>.md` (CLAUDE.md 7절).

## 문서 목록 (`docs/`)

| 문서 | 내용 |
|---|---|
| [Scope_Definition.md](docs/Scope_Definition.md) | 프로젝트 최초 요구사항(스코프, 분류 로직, AIPro+ 연동, Phase 로드맵) |
| [Architecture_Design.md](docs/Architecture_Design.md) | 전체 시스템 설계 — 모듈 구조, 데이터 흐름, 기술 스택, Docker 구성 |
| [P0_설계서_Common.md](docs/P0_설계서_Common.md) | **Phase 0** 공통 모듈(`config`/`constants`/`domain`/`exceptions`/`run_context`) 필드·시그니처 설계 |
| [P0_설계서_Logging.md](docs/P0_설계서_Logging.md) | **Phase 0** 로깅 표준(JSON 스키마, 라벨/본문 구분) — Loki/Grafana는 미연동, 향후 대비만 |
| [P0_테스트결과서_Common.md](docs/P0_테스트결과서_Common.md) | **Phase 0** 공통 모듈 테스트 실행 결과·등급별 커버리지 |
| [P1_DataGeneration_Requirements.md](docs/P1_DataGeneration_Requirements.md) | Phase 1(데이터 생성) 요구사항 정의서 — v0.2 재생성 포함 |
| [P1_Data_Preprocessing_Review.md](docs/P1_Data_Preprocessing_Review.md) | v0.1 데이터 결함 검토(CSV 이스케이프 오류, 분할 시드 미고정) |

## 진행 상황

현재 단계는 **Phase 0(범위 정의 + 공통 모듈 기반 구축)**이다. Phase 1(데이터 생성) 이후
코드는 아직 착수 전이며, 문서 파일명도 이 단계 산출물은 `P0_` 접두사를 붙여 구분한다
(CLAUDE.md 7절).

| 영역 | 상태 | 비고 |
|---|---|---|
| Scope/요구사항 정의 | 완료 | Scope_Definition.md |
| 전체 아키텍처 설계 | 완료 | Architecture_Design.md |
| **Phase 0** 공통 모듈 설계 | 완료 | P0_설계서_Common.md |
| **Phase 0** 로깅 표준 설계 | 완료 | P0_설계서_Logging.md |
| **Phase 0** 공통 모듈 코드+테스트(`config`/`constants`/`domain`/`exceptions`/`run_context`/`logging_config`) | 완료 | 23 tests passed, A+B 등급 커버리지 100% — P0_테스트결과서_Common.md, Docker(`docker/Dockerfile.pipeline`) 내부에서 실행 |
| **Phase 0** 임베딩 캐싱 설계 변경 | 완료 | MD5 해시 기반 레코드 중복 판별 제거 → `source` 필드에 분류 라벨값 저장, 콜렉션을 `<version>_<train\|test\|validation>`로 분리. 도메인(`DOMAIN_NAME`, 프로젝트 고정 1개)·콜렉션 모두 사전 등록 후에만 지식 데이터 등록 가능(둘 다 이미 존재하면 재등록하지 않음) — Scope_Definition.md 2.1절/Architecture_Design.md 참고 |
| Phase 1 요구사항 정의 | 완료 | v0.2 재생성 요구사항까지 정의됨, 설계서(P1_DataGeneration_Design)는 미작성 |
| Phase 1 데이터(v0.1) | 생성됨, 결함 있음 | `data/v0.1_from.Claude-Cowork/` — CSV 이스케이프 오류 1건, 분할 시드 미고정 (P1_Data_Preprocessing_Review 참고) |
| Phase 1~5 코드 | 미착수 | |
| Docker: `Dockerfile.pipeline` 뼈대 | 완료 | Phase 0 공통 모듈 테스트 실행용. 호스트가 사내 프록시 경유 환경이면 `docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --build-arg no_proxy=$no_proxy`로 프록시를 넘겨야 `pip install`이 성공함(자동 상속 안 됨) |
| Docker: `Dockerfile.inference`, `docker-compose.yml` | 미착수 | |
| Loki/Grafana 연동 | 미착수 | P0_설계서_Logging.md 7절에 향후 방침만 기록, 지금은 stdout 로그까지만 |

이 표는 작업이 진행될 때마다 갱신한다.
