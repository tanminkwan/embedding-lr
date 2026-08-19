# 설계서 — Phase 1 데이터 생성 (DataGeneration)

[[P1_DataGeneration_Requirements]](요구사항정의서)를 [[Architecture_Design]] 2절(모듈
구조)·3절(Workflow 규약)의 실제 모듈 시그니처로 구체화한 설계서. [[CLAUDE.md]] 3절
순서상 2단계 산출물이다.

대상 모듈: `data_generation/{prompt_loader,jsonl_repository}.py`, `dataset/{combine,split}.py`.
[[Architecture_Design]]은 데이터 생성을 **Phase 1**(프롬프트 → 역할별 원본)과 **Phase 1.5**
(조합/분할)로 나누는데, 본 설계서는 [[P1_DataGeneration_Requirements]]와 동일하게 둘 다
다룬다.

## 1. 범위와 설계 전제

- **Phase 1(데이터 생성)의 "LLM 호출"은 결정적 자동화 스크립트가 아니라, 사람 또는 코딩
  에이전트가 프롬프트를 가지고 수행하는 상호작용 과정**이다 — v0.1 데이터도 이렇게
  만들어졌다(`data/v0.1_from.Claude-Cowork/`). 따라서 코드가 책임지는 범위는 ①프롬프트
  렌더링(`prompt_loader.py`)과 ②생성 결과를 스키마에 맞게 저장(`jsonl_repository.py`)으로
  한정한다 — [[CLAUDE.md]] 2절 **C등급**(`data_generation/*`, 자동화 테스트 비강제) 분류와
  일치.
- **Phase 1.5(조합/분할)는 완전 결정적**이다 — `role_*.jsonl` 9개 파일이 이미 존재한다고
  가정하고 시작하며, 같은 입력에 항상 같은 출력을 낸다 — [[CLAUDE.md]] 2절 **A등급**
  (`dataset/combine`, `dataset/split`) 분류와 일치, 테스트 먼저 작성.
- **저장 형식의 계층 분리(DIP)**: 지금은 JSONL 하나뿐이지만, 파일 형식은 언제든 바뀔 수
  있다(예: 원본 LLM 출력이 다른 텍스트 형식으로 오거나, 저장소가 Parquet/DB로 바뀌는 경우).
  그래서 `dataset/combine.py`·`dataset/split.py`는 파일이나 포맷을 전혀 모르고
  **`list[QueryRecord]`(메모리 객체) 위에서만 동작**한다. 파일 I/O는 오직
  `domain/interfaces.DataRepository` Protocol(이미 [[P0_설계서_Common]] 4절에 정의됨)의
  구현체 하나(`data_generation/jsonl_repository.JsonlRepository`)에만 있다. 나중에 형식이
  바뀌면 `DataRepository`를 구현하는 클래스 하나만 새로 추가하면 되고, `combine.py`·
  `split.py`·이들을 호출하는 CLI 오케스트레이션 로직은 한 줄도 고치지 않는다
  ([[CLAUDE.md]] 1절 DIP 원칙 "데이터 로더를 대체 구현체로 바꿔도 호출부가 깨지지 않아야
  한다"의 실제 적용 사례).

```
                              ┌─ DataRepository (Protocol, domain/interfaces.py) ─┐
                              │  load(path) -> list[QueryRecord]                  │
                              │  save(records, path) -> None                     │
                              └───────────────▲───────────────────────────────────┘
                                               │ 구현
                              JsonlRepository ─┘  (지금은 이것 하나. 형식이 바뀌면 여기만 교체)
                                               │
        dataset/combine.py, dataset/split.py ─┴─ list[QueryRecord]만 알고 파일/포맷은 모름
```

## 2. `data_generation/prompt_loader.py`

```python
def render_prompt(common_path: str, role_path: str) -> str:
    """공통 프롬프트(common_it.md 또는 common_non_it.md) + 역할별 프롬프트(roles/0N_*.md)를
    이 순서로 이어붙여, LLM에 그대로 입력할 최종 프롬프트 문자열을 만든다."""
```

- 입력: `prompt/common_it.md`\|`prompt/common_non_it.md`, `prompt/roles/0N_*.md` 파일 경로.
- 출력: 두 파일 내용을 개행 2개로 이어붙인 문자열. 판단 로직 없이 파일을 읽어 합치기만
  하므로 사실상 결정적이지만, [[CLAUDE.md]] 2절 표는 `data_generation/*` 전체를 C등급으로
  분류하므로 테스트를 강제하지는 않는다(선택적으로 작성 가능).
- CLI 연결: `cli/run_phase1.py --role <01~09> --output-prompt <path>`가 이 함수를 호출해
  렌더링된 프롬프트를 파일로 저장 → 사람/에이전트가 이 프롬프트로 LLM에서 질의·응답 쌍을
  생성 → 생성 결과를 `QueryRecord` 목록으로 만들어 3절의 저장소에 넘긴다.

## 3. `data_generation/jsonl_repository.py` — `DataRepository` 구현체

```python
class JsonlRepository:
    """domain.interfaces.DataRepository 구현체. 현재 유일한 저장 형식(JSONL)을 담당."""

    def load(self, path: str) -> list[QueryRecord]:
        """파일을 한 줄씩 읽어 JSON 파싱 → QueryRecord로 검증. 파싱 실패 줄이 있으면
        어느 줄(1-indexed)인지 포함해 DataValidationError."""

    def save(self, records: list[QueryRecord], path: str) -> None:
        """QueryRecord 목록을 JSONL(레코드 1건 = 1줄)로 저장. 대상 경로가 이미 존재하면
        DataValidationError — 덮어쓰기 금지([[CLAUDE.md]] 5절 입출력 보존)."""
```

- `data/<version>/role_0N_<topic>.jsonl` 저장 시 각 줄은
  `{"질의": ..., "응답": ..., "카테고리": ...}` (키는 `constants.JSON_KEY_*`).
- 저장 전 검증: (a) `카테고리`가 `CLASS_LABELS`에 없으면 거부(`QueryRecord`의 pydantic
  validator가 이미 담당 — [[P0_설계서_Common]] 3절) (b) 대상 파일 기존 존재 시 거부.
- 이 클래스는 Phase 1(role 파일 저장)과 Phase 1.5(조합/분할 결과 파일 로드·저장) 양쪽에서
  **공유**된다 — 형식이 같으므로 저장소 구현체를 중복시키지 않는다.

## 4. `dataset/combine.py`

```python
def combine(role_records: list[list[QueryRecord]]) -> list[QueryRecord]:
    """role_01~09 9개 파일에서 이미 로드된 QueryRecord 목록 9개를 순서 무관하게 concat.
    클래스당 정확히 200건(총 1,000건)이 아니면 DataValidationError(클래스별 실제 건수를
    메시지에 포함)."""
```

- 입력은 **이미 로드된 객체**(`JsonlRepository.load()`의 반환값 9개)다 — 이 함수는 파일
  경로를 받지 않는다(1절의 DIP 원칙).
- 클래스별 건수 검증 실패 시 `DataValidationError`: 예) `"IT 199건 (기대 200건)"`처럼 어떤
  클래스가 몇 건 부족/초과한지 명시 — [[P1_Data_Preprocessing_Review]] 3.1절 사고(1건
  누락)를 조기에 잡기 위함.
- CLI 연결: `cli/run_phase1_5.py --input-dir data/<version> --output data/<version>/data.jsonl`가
  `JsonlRepository.load()` × 9 → `combine()` → `JsonlRepository.save()` 순서로 호출.

## 5. `dataset/split.py`

```python
def split(
    records: list[QueryRecord],
    seed: int = RANDOM_SEED,
    ratios: dict[str, int] = SPLIT_RATIOS,
) -> dict[str, list[QueryRecord]]:
    """클래스별 3:1:1 stratified 분할. 반환 키는 DATA_SPLITS(["train","test","validation"])와
    동일. 클래스별 결과 건수가 ratios 비율과 맞지 않으면 DataValidationError."""
```

- 알고리즘: 클래스별로 레코드를 그룹핑 → 그룹 내부에서 `seed`로 결정적 셔플 → `ratios`
  비율(3:1:1)로 잘라 `train`/`test`/`validation`에 각각 append. 클래스 처리 순서와 그룹 내
  셔플 순서 모두 고정되어야 재현성이 보장된다([[P1_DataGeneration_Requirements]] 5절
  "재현성" 요구사항).
- `seed`/`ratios` 기본값은 `constants.RANDOM_SEED`/`constants.SPLIT_RATIOS`([[P0_설계서_Common]]
  1절에 추가) — [[CLAUDE.md]] 4절에 따라 코드에 직접 쓰지 않고 상수를 참조한다.
- 반환 후 저장 시 파일명 매핑: `"validation"` 키 → `val.jsonl` (콜렉션명에서는 `validation`
  전체를 쓰고 파일명에서는 `val`로 축약하는 기존 규칙과 동일 — [[P0_설계서_Common]] 1절).
- CLI 연결: `cli/run_phase1_5.py`가 `combine()` 결과를 이어서 `split()`에 넣고,
  `JsonlRepository.save()`로 `train.jsonl`/`test.jsonl`/`val.jsonl` 3개 파일에 저장.

## 6. 신규 상수 (`constants.py`, 이미 반영됨)

```python
SPLIT_RATIOS = {"train": 3, "test": 1, "validation": 1}   # Scope_Definition 3.3절 3:1:1
RANDOM_SEED = 42                                           # 값 자체는 임의, 고정이 핵심
```

## 7. 데이터 흐름 요약

```
prompt/*.md ──(prompt_loader.render_prompt)──> 렌더링된 프롬프트
     ──(사람/에이전트 + LLM, 코드 범위 밖)──> 질의·응답 목록
     ──(QueryRecord 검증 + JsonlRepository.save)──> role_01~09_*.jsonl (9개)

role_01~09_*.jsonl ──(JsonlRepository.load × 9)──> list[QueryRecord] × 9
     ──(dataset.combine)──> data.jsonl (JsonlRepository.save)
     ──(JsonlRepository.load + dataset.split)──> train.jsonl / test.jsonl / val.jsonl
```

## 8. 테스트 등급 및 완료 기준

| 모듈 | 등급([[CLAUDE.md]] 2절) | 목표 커버리지 | 비고 |
|---|---|---|---|
| `data_generation/prompt_loader.py` | C | 측정 제외 | 자동화 테스트 비강제, 선택적 작성 가능 |
| `data_generation/jsonl_repository.py` | B | ≥ 70% | 파일 I/O 있음 — happy path + 파싱 실패/중복 저장 에러 케이스 |
| `dataset/combine.py` | A | ≥ 90% | 순수 로직 — 테스트 먼저, 클래스 불균형 케이스 포함 |
| `dataset/split.py` | A | ≥ 90% | 순수 로직 — 테스트 먼저, seed 고정 재현성 케이스 포함 |

완료 기준은 [[P1_DataGeneration_Requirements]] 7절과 동일하며, 산출물 확장자만 `.jsonl`
이다. 특히 "동일 입력으로 재실행 시 분할 결과가 바이트 단위로 동일"함은 `split()`의
`seed` 고정 여부를 직접 검증하는 테스트 케이스로 커버한다.

## 9. 관련 문서/코드

- 요구사항: [[P1_DataGeneration_Requirements]]
- 상위 설계: [[Architecture_Design]] 2절(모듈 구조), 3절(Workflow 규약)
- 공통 모듈: [[P0_설계서_Common]] 1절(`SPLIT_RATIOS`/`RANDOM_SEED`/`JSON_KEY_*`), 4절
  (`DataRepository` Protocol)
- 관련 코드(계획): `src/embedding_lr/data_generation/{prompt_loader,jsonl_repository}.py`,
  `src/embedding_lr/dataset/{combine,split}.py`, `src/embedding_lr/cli/{run_phase1,run_phase1_5}.py`
- 테스트(계획): `tests/unit/test_{combine,split}.py`, `tests/integration/test_jsonl_repository.py`
