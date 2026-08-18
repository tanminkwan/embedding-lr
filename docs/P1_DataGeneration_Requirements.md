# P1_DataGeneration_Requirements — Phase 1 요구사항 정의서

[[Scope_Definition]] 3절(학습 데이터 확보 전략)과 7절(Phase 로드맵)을 Phase 1 단위로
구체화한 요구사항 정의서. [[CLAUDE.md]] 3절 "작업 진행 순서"의 1단계 산출물이며, 다음
산출물은 [[P1_DataGeneration_Design]](설계서)에서 다룬다.

## 1. 목적 (Why)

실 운영 로그를 사용할 수 없는 환경에서, Phase 3(모델 학습)·Phase 4(검증)에 투입할
**5-class 라벨링된 질의-응답 합성 데이터셋**을 LLM으로 확보한다. 이 데이터가 이후 모든
Phase(임베딩 변환·학습·검증·추론)의 유일한 입력 소스이므로, 여기서 확정하는 스키마·품질
기준이 전체 파이프라인의 신뢰도를 좌우한다.

## 2. 배경 및 제약

- v0.1 데이터(`data/v0.1_from.Claude-Cowork/`)가 이미 생성되어 있으나,
  [[P1_Data_Preprocessing_Review]]에서 다음 문제가 발견됨:
  1. `role_03_network.csv`의 CSV 이스케이프 오류로 레코드 1건 손실 + 파싱 오류 행 6건 발생
  2. `data.csv`→`train/test/val.csv` 분할 시 랜덤 시드가 고정되어 있지 않아, 재생성할 때마다
     분할 결과(및 결손 데이터가 어느 split에 떨어지는지)가 달라짐 — 재현성 없음
- 본 문서는 이 두 문제를 해결한 **v0.2 재생성**을 포함하여, Phase 1의 요구사항을
  명문화한다.

## 3. 범위

### In Scope

| # | 항목 |
|---|---|
| 1 | IT 5개 역할(`prompt/roles/01~05`) × 40건 = 200건 프롬프트 기반 생성 |
| 2 | NON_IT 4개 하위카테고리(`prompt/roles/06~09`) × 200건 생성 |
| 3 | `role_01~09_*.csv` (역할별 원본, source of truth) 생성 |
| 4 | `role_01~09_*.csv` → `data.csv` 재조합 (concat) |
| 5 | `data.csv` → `train.csv`/`test.csv`/`val.csv` 클래스별 3:1:1 **시드 고정** 분할 |
| 6 | CSV RFC4180 이스케이프 규정 준수 (필드 내 `"`, `,`, 개행 처리) |

### Out of Scope (다른 Phase 책임)

| 항목 | 담당 Phase |
|---|---|
| 임베딩 변환, 마크다운 코드펜스 제거 | Phase 2 |
| 모델 학습·하이퍼파라미터 탐색 | Phase 3 |
| 정확도/F1 평가 | Phase 4 |

## 4. 기능 요구사항

### 4.1 데이터 생성 방식

- **IT** (`prompt/common_it.md` + `prompt/roles/01~05`): 미들웨어/OS/네트워크/DBA/DevOps
  5개 역할, 역할별 40건.
- **NON_IT** (`prompt/common_non_it.md` + `prompt/roles/06~09`): DAILY/KNOWLEDGE/CREATIVE/
  ANOMALY 4개 카테고리, 카테고리별 200건.
- 난이도 분포: 단순 커맨드/개념 50% + 실무 트러블슈팅 50% (IT 기준).
- 응답 스타일: 반말, 간결체, 코드블록 포함 가능, 본론만.

### 4.2 출력 스키마

| 컬럼 | 타입 | 허용값 |
|---|---|---|
| `질의` | string | 2~100자 (기존 데이터 실측 2~63자 참고) |
| `응답` | string | 8~1000자 (기존 데이터 실측 8~671자 참고) |
| `카테고리` | string (enum) | `IT`, `DAILY`, `KNOWLEDGE`, `CREATIVE`, `ANOMALY` 중 하나 |

### 4.3 재조합 및 분할

- `role_01~09_*.csv` 9개 파일을 **순서 무관 concat**하여 `data.csv` 생성 — 클래스당
  정확히 200건, 총 1,000건이어야 함.
- `data.csv`를 클래스별로 **3:1:1 비율**(120/40/40)로 분할하여 `train.csv`/`test.csv`/
  `val.csv` 생성. 분할은 랜덤이지만 **시드를 코드에 고정**하여 재실행 시 항상 동일한
  분할 결과가 나오도록 한다 (2절의 재현성 문제 해결).

## 5. 비기능 요구사항 (품질 기준)

| 항목 | 기준 |
|---|---|
| 결측값 | `질의`/`응답`/`카테고리` 공백·NULL 0건 |
| 중복 | (`질의`, `카테고리`) 조합 기준 중복 0건 |
| 인코딩 | UTF-8, BOM 없음 |
| 개행 | LF만 허용 (CRLF 금지) |
| CSV 이스케이프 | 필드 내 `"` 문자는 `""`로 이중 처리 — `role_03_network.csv` 사고 재발 방지 |
| 클래스 균형 | 클래스당 정확히 200건 (총 1,000건), `train`/`test`/`val` 각 120/40/40건 |
| 재현성 | 동일 입력(`role_*.csv`)으로 재실행 시 `data.csv`, `train/test/val.csv`가 바이트 단위로 동일해야 함(분할 시드 고정) |

## 6. 산출물

| 파일 | 설명 | 명명 규칙 근거 |
|---|---|---|
| `data/<version>/role_01~09_*.csv` | 역할/카테고리별 원본 (source of truth) | 기존 유지 |
| `data/<version>/data.csv` | 9개 role 파일 재조합본 (1,000건) | 기존 유지 |
| `data/<version>/train.csv` | 학습셋 (600건) | 기존 유지 |
| `data/<version>/test.csv` | 테스트셋 (200건) | 기존 유지 |
| `data/<version>/val.csv` | 검증셋 (200건) | 기존 유지 |

## 7. 완료 기준 (Acceptance Criteria)

- [ ] `role_01~09_*.csv` 9개 파일 각각 정상 파싱(파싱 오류 행 0건), 각 200건(IT는 역할당 40건)
- [ ] `data.csv` 1,000건, 클래스당 정확히 200건, 파싱 오류 행 0건
- [ ] `train.csv`(600)/`test.csv`(200)/`val.csv`(200) 클래스당 120/40/40건, 파싱 오류 행 0건
- [ ] 5절의 비기능 요구사항(결측/중복/인코딩/개행/이스케이프/재현성) 전항목 통과
- [ ] 동일 입력으로 재실행 시 분할 결과 동일함을 확인(시드 고정 검증)

## 8. 리스크 및 참고사항

- `role_03_network.csv`의 이스케이프 오류는 **원본에서부터 수정**해야 하며, `data.csv`나
  `test.csv`를 직접 수정하면 다음 재조합 시 수정이 사라진다 —
  [[P1_Data_Preprocessing_Review]] 3.1절 및 [[CLAUDE.md]] 5절(입출력 보존 원칙) 참고.
- 마크다운 코드펜스(\`\`\`)는 IT 응답의 약 38.7%에 존재하며, 이는 Phase 1의 정제 대상이
  아니다(본문 명령어는 의미 있는 신호이므로 원문 그대로 보존). 코드펜스 구분자 제거는
  Phase 2 임베딩 입력 생성 시 처리한다 — [[P1_Data_Preprocessing_Review]] 3.2절 참고.
