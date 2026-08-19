# P1_DataPreparation_Requirements — Phase 1 요구사항 정의서

[[Scope_Definition]] 3절(학습 데이터 확보 전략)과 7절(Phase 로드맵)을 Phase 1 단위로
구체화한 요구사항 정의서. [[CLAUDE.md]] 3절 "작업 진행 순서"의 1단계 산출물이며, 다음
산출물은 [[P1_설계서_DataPreparation]](설계서)에서 다룬다.

**Phase 1은 새 데이터를 만드는 단계가 아니다.** 질의·응답 콘텐츠 자체는 LLM 프롬프트
기반으로 이미 확보되어 있다(현재는 CSV, `data/v0.1_from.Claude-Cowork/role_*.csv`) —
그 콘텐츠를 만드는 과정(프롬프트 설계, LLM과의 상호작용)은 이 저장소의 코드가 하는 일이
아니다([[Scope_Definition]] 3절 참고). Phase 1 코드가 다루는 범위는 **이미 확보된 원본을
학습 파이프라인이 쓰는 형식(JSONL)으로 변환하고, 재조합·분할하여 학습 가능한 상태로
준비하는 것**이다.

## 1. 목적 (Why)

Phase 3(모델 학습)·Phase 4(검증)에 투입할 `train.jsonl`/`test.jsonl`/`val.jsonl`을
만든다. 이 데이터가 이후 모든 Phase(임베딩 변환·학습·검증·추론)의 유일한 입력 소스이므로,
여기서 확정하는 변환·검증 기준이 전체 파이프라인의 신뢰도를 좌우한다.

## 2. 배경 및 제약

- v0.1 데이터(`data/v0.1_from.Claude-Cowork/`, CSV 포맷)가 이미 확보되어 있으나,
  [[P1_Data_Preprocessing_Review]]에서 다음 문제가 발견됨:
  1. `role_03_network.csv`의 CSV 이스케이프 오류로 레코드 1건 손실 + 파싱 오류 행 6건 발생
     — 응답 텍스트에 포함된 `"` 문자를 CSV가 제대로 이스케이프하지 못해 발생한 구조적
     문제. **원본에서 직접 수정 완료**(`""`로 이중 이스케이프) — [[P1_Data_Preprocessing_Review]]
     5절 참고. v0.2부터는 이 이스케이프 규칙 자체가 필요 없는 **JSONL**로 출력 포맷을
     전환해 이 문제 유형을 원천 차단한다([[Scope_Definition]] 3.3절 참고).
  2. `data.csv`→`train/test/val.csv` 분할 시 랜덤 시드가 고정되어 있지 않아, 재생성할 때마다
     분할 결과(및 결손 데이터가 어느 split에 떨어지는지)가 달라짐 — 재현성 없음. `dataset.split`의
     `seed` 고정으로 해결.
- 본 문서는 이 두 문제를 해결한 **v0.2 변환**(CSV→JSONL 전환 포함)을 포함하여, Phase 1의
  요구사항을 명문화한다.
- **원본 형식이 CSV라는 사실은 고정 전제가 아니다.** 다음에 확보되는 데이터가 다른 형식으로
  올 수 있으므로, 원본 로딩은 `DataRepository` Protocol 뒤로 추상화한다
  ([[P1_설계서_DataPreparation]] 참고) — 이번 CSV는 `CsvRepository`라는 구현체 하나일 뿐이다.

## 3. 범위

### In Scope

| # | 항목 |
|---|---|
| 1 | 이미 확보된 원본(`data/<version>/role_01~09_*.csv`) 로딩 — `DataRepository` 구현체(`CsvRepository`) |
| 2 | 로딩한 레코드를 JSONL로 저장 — `role_01~09_*.jsonl` (`JsonlRepository`) |
| 3 | `role_01~09_*.jsonl` → `data.jsonl` 재조합 (concat), 클래스당 정확히 200건 검증 |
| 4 | `data.jsonl` → `train.jsonl`/`test.jsonl`/`val.jsonl` 클래스별 3:1:1 **시드 고정** 분할 |
| 5 | JSONL 무결성 준수 — 매 줄이 정확히 레코드 1건에 대응하는 유효한 JSON 객체(응답 내 개행은 JSON 문자열 이스케이프 `\n`으로 표현되어 줄 경계를 깨지 않아야 함) |

### Out of Scope (다른 Phase 또는 코드 밖 책임)

| 항목 | 담당 |
|---|---|
| 질의·응답 콘텐츠 생성(LLM 프롬프트 기반) | 코드 밖(사람/에이전트가 프롬프트로 직접 수행) — [[Scope_Definition]] 3절 |
| 임베딩 변환, 마크다운 코드펜스 제거 등 텍스트 정제 | Phase 2 ([[P2_설계서_TextCleaning]]) |
| 모델 학습·하이퍼파라미터 탐색 | Phase 3 |
| 정확도/F1 평가 | Phase 4 |

## 4. 기능 요구사항

### 4.1 원본 스키마 (입력 — 현재 CSV)

| 컬럼 | 타입 | 허용값 |
|---|---|---|
| `질의` | string | 2~100자 (기존 데이터 실측 2~63자 참고) |
| `응답` | string | 8~1000자 (기존 데이터 실측 8~671자 참고), 멀티라인 허용(CSV 인용 필드) |
| `카테고리` | string (enum) | `IT`, `DAILY`, `KNOWLEDGE`, `CREATIVE`, `ANOMALY` 중 하나 |

### 4.2 출력 스키마 (JSONL)

각 줄은 아래 3개 키를 가진 JSON 객체 1건이다(입력과 키 이름 동일, 포맷만 CSV→JSON).

| 키 | 타입 | 허용값 |
|---|---|---|
| `질의` | string | 4.1과 동일 |
| `응답` | string | 4.1과 동일, 멀티라인인 경우 JSON 문자열 이스케이프(`\n`)로 표현 |
| `카테고리` | string (enum) | 4.1과 동일 |

예: `{"질의": "nginx 재시작 방법?", "응답": "systemctl restart nginx", "카테고리": "IT"}`

### 4.3 재조합 및 분할

- `role_01~09_*.jsonl` 9개 파일을 **순서 무관 concat**(라인 단위 이어붙이기)하여
  `data.jsonl` 생성 — 클래스당 정확히 200건, 총 1,000건이어야 함.
- `data.jsonl`를 클래스별로 **3:1:1 비율**(120/40/40)로 분할하여 `train.jsonl`/`test.jsonl`/
  `val.jsonl` 생성. 분할은 랜덤이지만 **시드를 코드에 고정**하여 재실행 시 항상 동일한
  분할 결과가 나오도록 한다 (2절의 재현성 문제 해결).

## 5. 비기능 요구사항 (품질 기준)

| 항목 | 기준 |
|---|---|
| 결측값 | `질의`/`응답`/`카테고리` 공백·NULL 0건 |
| 중복 | (`질의`, `카테고리`) 조합 기준 중복 0건 |
| 인코딩 | UTF-8, BOM 없음 |
| 개행(파일 구분자) | JSONL 레코드 구분은 LF만 허용(CRLF 금지) — 파일 내 물리적 줄 수 = 레코드 수와 항상 일치해야 함 |
| JSON 유효성 | 매 줄이 표준 `json` 라이브러리로 파싱 가능한 객체 1건이어야 함 — `role_03_network.csv` 사고(2절, CSV 이스케이프 오류)와 같은 유형의 파싱 실패 원천 차단(JSON 표준 이스케이프는 언어 표준 라이브러리가 처리하므로 수기 이스케이프 규칙 불필요) |
| 클래스 균형 | 클래스당 정확히 200건 (총 1,000건), `train`/`test`/`val` 각 120/40/40건 |
| 재현성 | 동일 입력(`role_*.jsonl`)으로 재실행 시 `data.jsonl`, `train/test/val.jsonl`가 바이트 단위로 동일해야 함(분할 시드 고정) |

## 6. 산출물

| 파일 | 설명 | 비고 |
|---|---|---|
| `data/<version>/role_01~09_*.csv` | 이미 확보된 원본 (source of truth, 입력) | v0.1은 CSV — 형식은 향후 바뀔 수 있음 |
| `data/<version>/role_01~09_*.jsonl` | 원본을 JSONL로 변환한 결과 | v0.2부터 신규 |
| `data/<version>/data.jsonl` | 9개 role 파일 재조합본 (1,000건) | v0.2부터 신규 |
| `data/<version>/train.jsonl` | 학습셋 (600건) | v0.2부터 신규 |
| `data/<version>/test.jsonl` | 테스트셋 (200건) | v0.2부터 신규 |
| `data/<version>/val.jsonl` | 검증셋 (200건) | v0.2부터 신규 |

## 7. 완료 기준 (Acceptance Criteria)

- [x] `role_03_network.csv`의 CSV 이스케이프 오류 원본 수정 완료(6개 파싱 오류 행 → 0건)
- [x] `role_01~09_*.jsonl` 9개 파일 각각 정상 변환(JSON 파싱 실패 행 0건), 각 200건(IT는 역할당 40건)
- [x] `data.jsonl` 1,000건, 클래스당 정확히 200건, JSON 파싱 실패 행 0건
- [x] `train.jsonl`(600)/`test.jsonl`(200)/`val.jsonl`(200) 클래스당 120/40/40건, JSON 파싱 실패 행 0건
- [x] 5절의 비기능 요구사항(결측/중복/인코딩/개행/JSON 유효성/재현성) 전항목 통과
- [x] 동일 입력으로 재실행 시 분할 결과 동일함을 확인(시드 고정 검증, `dataset/test_split.py`)

`data/v0.2/`에 대해 위 항목 모두 실제로 실행·검증 완료 — [[P1_설계서_DataPreparation]] 참고.

## 8. 리스크 및 참고사항

- `role_03_network.csv`의 이스케이프 오류는 v0.1(CSV 포맷)에서 발생한 사고이며, 원본에서
  직접 수정했다([[P1_Data_Preprocessing_Review]] 5절). v0.2부터는 JSONL 전환으로 이 유형의
  파싱 사고 자체가 구조적으로 재발할 수 없다. 원본이 하류 결과에 의해 덮어써지지 않는다는
  원칙은 포맷과 무관하게 동일하게 적용된다 — `role_01~09_*.csv`(원본)/`role_01~09_*.jsonl`
  (변환본)가 source of truth이고, `data.jsonl`나 `test.jsonl`를 직접 수정하면 다음
  재조합/재분할 시 수정 내용이 사라진다 — [[CLAUDE.md]] 5절(입출력 보존 원칙) 참고.
- 마크다운 코드펜스(\`\`\`)는 IT 응답의 약 38.7%에 존재하며, 이는 Phase 1의 정제 대상이
  아니다(본문 명령어는 의미 있는 신호이므로 원문 그대로 보존). 코드펜스 구분자 제거는
  Phase 2 임베딩 입력 생성 시 처리한다 — [[P1_Data_Preprocessing_Review]] 3.2절,
  [[P2_설계서_TextCleaning]] 참고.
