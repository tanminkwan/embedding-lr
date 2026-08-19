# 설계서 — Logging

[[Architecture_Design]] 3절(Workflow 친화 규약)과 5절(기술 스택: "로깅 | 표준 logging
(JSON 포맷)")을 구체화하는 문서. 모든 Phase(1~5) 공용 로깅 표준을 정의한다.
[[CLAUDE.md]] 4절(하드코딩 금지) 및 5절(모니터링·재시작 가능) 원칙을 따른다.

## 1. 목적 및 범위

- 코딩 시작 전, "로그를 어떻게 남길지"만 먼저 확정한다. **로그 수집/전송 인프라(Loki,
  Grafana Alloy 등)는 아직 붙이지 않는다** — 지금은 stdout 출력까지만 표준화하고,
  향후 Loki를 연결할 때 애플리케이션 코드를 변경하지 않아도 되도록 스키마를 미리
  고정해 둔다.
- 적용 대상: `cli/run_phase1.py` ~ `run_phase4.py`, `inference/api.py` 등 모든
  실행 진입점, 그리고 그 하위에서 호출되는 모듈(`embedding.pipeline`,
  `training.trainer` 등).

## 2. 로그 출력 방식

- 표준 `logging` 모듈 + JSON 포맷터(`python-json-logger`)를 사용한다.
- 로그는 **stdout에만** 출력한다. 파일로 직접 쓰지 않는다 — Docker가 stdout을
  캡처하는 것으로 충분하며, 그 이후 목적지(Loki 등)를 결정하는 것은 애플리케이션이
  아니라 인프라(compose/collector 설정) 책임이다. 이는 DIP 취지와 동일하게, 로깅
  "발생부"와 로깅 "수집부"를 분리해 둔다.
- `status/<phase>_<run_id>.json`([[Architecture_Design]] 3절)는 로그와 별개로,
  Phase 실행의 시작/종료/성공/실패 요약만 담는 상태 파일이다. 로그는 실행 중 발생하는
  모든 이벤트의 상세 스트림이고, 상태 파일은 그 실행의 최종 요약이다 — 둘은 동일한
  `run_id`로 교차 조회 가능해야 한다(3절 참고).

## 3. 로그 라인 스키마

모든 로그 라인은 아래 필드를 포함한다.

| 필드 | 타입 | 예시 | 설명 |
|---|---|---|---|
| `timestamp` | string (ISO8601, UTC) | `2026-08-19T05:12:33.482Z` | 발생 시각 |
| `level` | string | `INFO`, `WARNING`, `ERROR` | 로그 레벨 |
| `service` | string | `embedding_lr` | 서비스/프로젝트 식별자, 고정값 |
| `phase` | string | `embedding`, `training`, `inference` | 실행 중인 Phase/서비스 |
| `logger` | string | `embedding.pipeline` | 로거 이름(모듈 경로) — `logging.getLogger(__name__)` |
| `run_id` | string | `20260819-051233-a1b2` | 실행 1회당 고유값. `status/<phase>_<run_id>.json`과 **동일 값**을 재사용 |
| `message` | string | `"upsert 완료: 128건"` | 사람이 읽는 메시지 |
| `extra` | object (선택) | `{"error": "...", "record_count": 128}` | 에러 스택, 처리 건수 등 가변 컨텍스트 |

예시:

```json
{"timestamp": "2026-08-19T05:12:33.482Z", "level": "INFO", "service": "embedding_lr", "phase": "embedding", "logger": "embedding.pipeline", "run_id": "20260819-051233-a1b2", "message": "upsert 완료: 128건", "extra": {"record_count": 128}}
```

## 4. 라벨 vs 본문 구분 원칙 (향후 Loki 연동 대비)

Loki를 붙일 경우를 대비해, 필드를 값의 카디널리티 기준으로 미리 구분해 둔다.
지금 코드에서 별도로 처리할 것은 없지만, 이 구분에 맞춰 필드를 추가/변경한다.

| 구분 | 필드 | 이유 |
|---|---|---|
| 라벨 후보(저카디널리티) | `service`, `phase`, `env` | 값의 종류가 몇 개로 고정됨 — Loki 라벨/인덱스로 써도 안전 |
| 본문 전용(고카디널리티) | `run_id`, `message`, `logger`, `extra` | 실행마다 값이 달라짐 — 라벨로 쓰면 Loki 스트림이 과다 생성되어 인덱스가 터짐. JSON 본문에만 두고 LogQL `| json` 파싱으로 필터링 |

**규칙**: 새 필드를 추가할 때, 그 값이 "실행마다/레코드마다 달라지는가"를 기준으로
본문 전용 여부를 판단한다. 애매하면 본문 전용으로 분류한다(라벨 오분류의 비용이
더 크다).

## 5. 설정

- `LOG_LEVEL`은 `.env`에서 읽는다([[CLAUDE.md]] 4절 — 하드코딩 금지). 기본값은
  `INFO`.
- `service`, `env` 같은 고정 라벨 후보 값도 `.env`에서 주입한다(예: `ENV=local`,
  `ENV=prod`) — 코드에 직접 쓰지 않는다.
- 도메인 상수가 아니라 실행 환경마다 달라지는 값이므로 `constants.py`가 아니라
  `.env` + `config.py`(pydantic-settings)로 로딩한다.

## 6. 코드 구조 (계획 — 미구현)

```
src/embedding_lr/
└── logging_config.py   # setup_logging() 단일 함수 — SRP
```

- `setup_logging()`: `LOG_LEVEL`, `service` 값을 읽어 stdout 핸들러 + JSON
  포맷터를 구성한 표준 `logging` 설정을 1회 적용한다.
- 각 `cli/run_phaseN.py`, `inference/api.py`는 진입 시 `setup_logging()`을
  1회 호출한 뒤 `logging.getLogger(__name__)`으로 로거를 얻어 사용한다.
- 이 모듈 자체는 결정적 순수 로직이 아니라 설정/초기화이므로, [[CLAUDE.md]] 2절
  등급 분류상 **B등급**(오케스트레이션)에 가깝다 — 구현 후 통합 테스트(핸들러/포맷터가
  올바르게 붙는지, JSON이 스키마대로 직렬화되는지)로 충분하다.

## 7. 향후 Loki/Grafana 연동 시 변경 범위 (미착수, 참고용)

지금은 구현하지 않는다. 나중에 필요해지면 아래만 추가하면 되고, 앱 코드
(`logging_config.py` 및 로그 호출부)는 변경하지 않는 것이 목표다.

- `docker-compose.yml`의 각 서비스에 로그 드라이버(예: Docker `loki` 드라이버) 설정
  추가, 또는 Grafana Alloy를 별도 컨테이너로 추가해 컨테이너 stdout을 스크레이핑.
- Loki 엔드포인트 등 연결 정보는 `.env`/compose 환경변수로만 주입(하드코딩 금지 동일 적용).
- Grafana 대시보드/알림 구성은 이 프로젝트 코드 범위 밖(인프라 레포 또는 별도 설정)으로 분리.

## 8. 관련 문서/코드

- 상위 설계: [[Architecture_Design]] 3절(Workflow 규약), 5절(기술 스택)
- 관련 코드(계획): `src/embedding_lr/logging_config.py`, `src/embedding_lr/config.py`
- 상태 파일 규약: [[Architecture_Design]] 3절 `status/<phase>_<run_id>.json`
