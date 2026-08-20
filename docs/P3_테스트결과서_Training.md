# 테스트결과서 — Phase 3 모델 학습 (Training)

[[P3_설계서_Training]]에 따라 구현한 `domain/models.py`(갱신), `training/{trainer,
persistence}.py`, `cli/run_phase3.py`의 테스트 결과를 기록한다. [[CLAUDE.md]] 6절에
따라 **모든 실행은 호스트가 아니라 Docker 컨테이너 내부에서** 이뤄졌다. AIPro+/
Embedding Service 등 외부 서비스는 이 Phase에서 아예 호출하지 않는다([[P3_요구사항정의서_Training]]
3절 Out of Scope) — `training/trainer.py`가 읽는 입력은 Phase 2가 만든 로컬
`*_vectors.parquet` 뿐이다.

이 문서는 성격이 다른 두 검증을 구분해서 기록한다.

- **1~3절 — 기능 테스트(자동화 `pytest` 스위트)**: `tests/unit`, `tests/integration`
  아래 자동화 테스트. 대부분 저차원(4차원) **합성 fixture 데이터**로 로직/분기를
  빠르고 결정적으로 검증한다(fake `Classifier`가 아니라 진짜 scikit-learn을 쓰지만,
  입력 데이터 자체는 합성). CI에서 매번 재실행되는 회귀 방지용 테스트다.
- **4절 — 통합 테스트(실데이터 End-to-End)**: 자동화 스위트에는 포함되지 않은,
  Phase 2가 실제로 만든 `data/v0.2/{train,test}_vectors.parquet`(1024D, 5-class,
  600/200건)을 그대로 컨테이너에 마운트해 `cli/run_phase3.main()`을 직접 실행한
  1회성 검증이다. 자동화 fixture로는 드러나지 않던 실제 버그(6절)를 여기서 발견했다.

## 1. 기능 테스트 — 실행 방법

```bash
docker build -f docker/Dockerfile.pipeline \
  --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy \
  --build-arg no_proxy=$no_proxy \
  -t embedding-lr-pipeline:phase3 .

# Phase 3 범위만
docker run --rm embedding-lr-pipeline:phase3 python -m pytest -q \
  tests/unit/test_domain_models.py \
  tests/integration/test_trainer.py tests/integration/test_persistence.py \
  tests/integration/test_run_phase3.py \
  --cov=embedding_lr.training --cov=embedding_lr.cli.run_phase3 \
  --cov=embedding_lr.domain.models --cov-report=term-missing

# 프로젝트 전체(회귀 확인)
docker run --rm embedding-lr-pipeline:phase3 python -m pytest -q \
  --cov=embedding_lr --cov-report=term-missing
```

## 2. 기능 테스트 결과 요약

```
Phase 3 범위: 35 passed in 3.98s
프로젝트 전체: 121 passed in 6.69s
```

Phase 3 범위(A+B 등급 전체) 커버리지: 176 lines 중 175 covered = **99%**(프로젝트 목표
≥ 80%, 등급 B 목표 ≥ 70% 모두 충족). 프로젝트 전체 커버리지는 614 lines 중 606 covered
= **99%**.

| 모듈 | 등급([[CLAUDE.md]] 2절) | 목표 커버리지 | 실측 커버리지 | 비고 |
|---|---|---|---|---|
| `domain/models.py`(`HyperparamTrial`/`HyperparamSearchResult` 추가분 포함) | A | ≥ 90% | 100% (80 lines) | 필드 검증만 있는 순수 모델 — 정상 생성/필수 필드 누락/빈 `trials` 리스트 케이스 |
| `training/trainer.py` | B | ≥ 70% | 100% (49 lines) | `load_vectors` 스키마 오류 2케이스, `LogisticRegressionClassifier` predict_proba, tie-break(F1 동률 시 Accuracy) 포함 4케이스, `train_final_model` |
| `training/persistence.py` | B | ≥ 70% | 100% (17 lines) | save/load 왕복, 기존 파일 존재 시 실패(모델/탐색이력 각각), `ModelNotFoundError` |
| `cli/run_phase3.py` | B | ≥ 70% | 97% (30 lines, 1 miss: L59 `if __name__ == "__main__"`) | fixture parquet + fixture config JSON으로 E2E 실행, 상태 파일(`succeeded`) 확인 |

`if __name__ == "__main__":` 1줄은 다른 CLI 스크립트([[P2_테스트결과서_Embedding]] 2절)와
동일하게 표준 관용구라 커버리지 측정에서 실질적 의미가 없다.

## 3. 기능 테스트 케이스 상세

| 파일 | 케이스 수 | 확인 내용 |
|---|---|---|
| `tests/unit/test_domain_models.py`(신규 케이스만) | 5 | `HyperparamTrial` 정상 생성/필수 필드 누락, `HyperparamSearchResult` 정상 생성(trials 포함)/빈 trials 허용/필수 필드 누락 |
| `tests/integration/test_trainer.py` | 11 | `load_vectors` 정상 반환/컬럼 누락 거부/알 수 없는 label 거부(3), `LogisticRegressionClassifier.predict_proba`가 라벨명 dict 반환(1), `search_hyperparameters` 그리드 전체 조합 커버 + F1-macro 동률 시 Accuracy로 tie-break + NaN 점수(liblinear류 실패 조합)를 최하위로 처리 + 전 조합 실패 시 `DataValidationError`(4), `train_final_model` 반환값 검증(1) |
| `tests/integration/test_persistence.py` | 5 | 모델 save/load 왕복, 모델 저장 경로 기존 존재 시 `DataValidationError`, 모델 파일 없을 때 `ModelNotFoundError`, 탐색 이력 저장 내용 일치, 탐색 이력 경로 기존 존재 시 `DataValidationError` |
| `tests/integration/test_run_phase3.py` | 2 | CLI E2E(fixture parquet+config, 저차원 합성 데이터) — 모델/탐색이력 파일 생성 및 재로드 가능 확인, `status/phase3_*.json`에 `succeeded` 기록 |

이 4개 파일은 모두 합성 데이터(2-class, 4차원)를 쓴다 — 목적이 "분기·예외 로직이
맞는가"이지 "실데이터에서 모델이 잘 학습되는가"가 아니기 때문이다. 후자는 4절에서
별도로 검증했다.

## 4. 통합 테스트 — 실데이터 End-to-End 상세 결과

### 4.1 실행 방법

```bash
mkdir -p <out_dir>/status

docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "<out_dir>:/app/out" \
  -e AIPRO_BASE_URL=http://localhost:28000 \
  -e AIPRO_API_TOKEN=dummy \
  -e EMBEDDING_SERVER_BASE_URL=http://localhost:8000 \
  -e MODEL_DIR=/app/out \
  -e STATUS_DIR=/app/out/status \
  embedding-lr-pipeline:phase3 \
  python -m embedding_lr.cli.run_phase3 \
    --train data/v0.2/train_vectors.parquet \
    --test data/v0.2/test_vectors.parquet \
    --model-output /app/out/model.pkl \
    --search-output /app/out/hyperparams.json
```

`<out_dir>`은 컨테이너 밖 임시 출력 디렉터리를 가리키며, 산출물(`model.pkl`/
`hyperparams.json`/`status/phase3_*.json`)이 그 아래에 생성된다. `--config`를
생략해 기본값 `config/hyperparams_default.json`(이미지에 COPY됨, `C`×4 · `solver`×1
· `max_iter`×3 = 12조합)을 그대로 썼다. `AIPRO_*`/`EMBEDDING_SERVER_BASE_URL`은
`Settings()` 필수 필드 통과용으로만 존재하며, 이 CLI는 실제로 그 두 서비스를
호출하지 않는다.

### 4.2 입력 데이터 확인

`cli/run_phase3.main()`을 실행하기 전, 입력이 요구사항([[P3_요구사항정의서_Training]]
4.1절)과 일치하는지 컨테이너 안에서 직접 확인했다.

| 파일 | shape | 클래스별 건수 | embedding 차원 |
|---|---|---|---|
| `train_vectors.parquet` | (600, 2) | `IT`/`DAILY`/`KNOWLEDGE`/`CREATIVE`/`ANOMALY` 각 120건 | 1024 |
| `test_vectors.parquet` | (200, 2) | 5-class 각 40건 | 1024 |

### 4.3 실행 결과 — 상태/타이밍/산출물

| 항목 | 값 |
|---|---|
| `run_id` | `20260820-012615-329a` |
| 시작 | `2026-08-20T01:26:15.875Z` |
| 종료 | `2026-08-20T01:26:23.643Z` |
| 소요 시간 | 약 7.77초 (12조합 탐색 + train 재학습 포함) |
| `status/phase3_*.json` | `"status": "succeeded"`, `"error": null` |
| `model.pkl` | 42,097 bytes, joblib 재로드 성공 |
| `hyperparams.json` | 2,227 bytes, 12개 `trials` 전부 기록(누락 없음) |

### 4.4 하이퍼파라미터 탐색 전체 결과(12조합)

| C | max_iter | solver | accuracy | f1_macro |
|---|---|---|---|---|
| **10.0** | **500** | **lbfgs** | **0.995** | **0.99500** ← 선정(best) |
| 10.0 | 1000 | lbfgs | 0.995 | 0.99500 |
| 10.0 | 2000 | lbfgs | 0.995 | 0.99500 |
| 1.0 | 500 | lbfgs | 0.990 | 0.99000 |
| 1.0 | 1000 | lbfgs | 0.990 | 0.99000 |
| 1.0 | 2000 | lbfgs | 0.990 | 0.99000 |
| 0.1 | 500 | lbfgs | 0.975 | 0.97484 |
| 0.1 | 1000 | lbfgs | 0.975 | 0.97484 |
| 0.1 | 2000 | lbfgs | 0.975 | 0.97484 |
| 0.01 | 500 | lbfgs | 0.970 | 0.96998 |
| 0.01 | 1000 | lbfgs | 0.970 | 0.96998 |
| 0.01 | 2000 | lbfgs | 0.970 | 0.96998 |

`max_iter`는 이 데이터 규모(600건, 1024D)에서 세 값 모두 동일 수렴 결과를 내
경향에 영향이 없었다 — `C`가 클수록(정규화 약할수록) test set 점수가 단조 증가.
동률(F1-macro 완전 동일)이 3그룹(C=10.0/1.0/0.1/0.01 각각 max_iter 3종) 있었는데,
tie-break 2차 기준인 Accuracy도 동일해 그룹 내에서는 `GridSearchCV.cv_results_`가
반환한 순서상 첫 조합(`max_iter=500`)이 선정됐다 — 요구사항이 요구하는 것은
"F1-macro→Accuracy" 2단계 tie-break뿐이고 그 이상의 3차 기준은 정의돼 있지 않으므로
사양대로다.

### 4.5 최종 모델 독립 재검증(파이프라인 코드 밖에서 재확인)

`search_hyperparameters()`가 보고한 점수를 그대로 믿지 않고, 저장된 `model.pkl`을
다시 로드해 **파이프라인 코드와 무관하게** `sklearn.metrics`로 test set을 재평가했다:

```python
from embedding_lr.training.persistence import load_model
model = load_model("/app/out/model.pkl")
y_pred = [max(p, key=p.get) for p in model.predict_proba(X_test)]
# accuracy_score, f1_score(average="macro") 로 독립 계산
```

결과 — **독립 계산값이 탐색 시점 보고값과 정확히 일치**:

- accuracy = 0.995 (200건 중 199건 정답)
- f1_macro = 0.9949992186279106

클래스별 `classification_report`:

| 클래스 | precision | recall | f1-score | support |
|---|---|---|---|---|
| ANOMALY | 1.00 | 1.00 | 1.00 | 40 |
| CREATIVE | 1.00 | 1.00 | 1.00 | 40 |
| DAILY | 1.00 | 1.00 | 1.00 | 40 |
| IT | 1.00 | 0.97 | 0.99 | 40 |
| KNOWLEDGE | 0.98 | 1.00 | 0.99 | 40 |

유일한 오분류 1건은 실제 `IT`를 `KNOWLEDGE`로 예측한 케이스로 추정된다(IT recall
0.97=39/40, KNOWLEDGE precision 0.98=39/40인 패턴과 일치).

### 4.6 `Classifier` Protocol 계약 및 부수 확인

- `predict_proba()`가 반환한 dict의 값 합은 1.0(부동소수점 오차 내, 예:
  `0.9999999999999999`) — 정상적인 확률 분포.
- **사소한 관찰**: dict의 키는 `model.classes_`(numpy 배열)에서 그대로 가져와
  `numpy.str_` 타입이다(`np.str_`는 `str`의 서브클래스라 `==`/딕셔너리 키/JSON
  직렬화 모두 순수 `str`처럼 동작해 현재 기능상 문제는 없다). 합성 fixture 테스트는
  `set(...) == {"IT", "DAILY"}`처럼 값 비교만 해서 이 차이를 못 잡았고, 실데이터로
  타입을 직접 찍어봐서 발견했다. 당장 고칠 필요는 없지만 Phase 5(`inference/predictor.py`)
  구현 시 `PredictionResult.probabilities` 검증에서 걸리는 게 없는지 유의할 것.
- `data/v0.2/val_vectors.parquet`의 mtime을 실행 전후로 비교해 **변경되지 않았음**을
  확인 — Phase 3가 validation set을 전혀 읽지 않는다는 요구사항([[P3_요구사항정의서_Training]]
  2절)이 실제로 지켜짐을 재확인했다(`1787133729` → `1787133729`, 동일).

## 5. 완료 기준([[P3_요구사항정의서_Training]] 7절) 충족 확인

- [x] `train_vectors.parquet`(600건)으로 학습, `test_vectors.parquet`(200건)으로 하이퍼파라미터별 Accuracy/F1 산출 — 4.4절 12조합 전체 표
- [x] 탐색 범위(`C`/`solver`/`max_iter`)가 코드 수정 없이 설정 변경만으로 조정 가능함을 확인 — `config/hyperparams_default.json` 값만 바꿔 재확인(6절, liblinear 제거)
- [x] 최적 조합으로 학습한 모델이 `.pkl`로 저장되고 joblib으로 재로드 가능함을 확인 — 4.3/4.5절
- [x] 탐색 이력 파일에 모든 시도 조합 + 지표가 기록됨을 확인(누락 없음) — 4.4절
- [x] `Classifier` Protocol 계약(`fit`/`predict_proba`) 준수 확인(`predict_proba`가 라벨명 dict 반환) — 4.6절
- [x] Docker 컨테이너 내부에서 학습 스크립트+테스트 실행 확인 — 1절/4.1절
- [x] 등급 B 커버리지 ≥ 70% 확인(2절 표, 실측 97~100%)

## 6. 재작업 내역

구현 중 설계서 작성 시점(2026-08-19)과 실제 설치된 `scikit-learn`(고정 범위 `>=1.4`,
실제 해석 버전 1.9.0) 사이의 API 차이로 두 차례 재작업이 있었다. 둘 다 **기능
테스트(합성 데이터)만으로는 드러나지 않고, 4절의 실데이터 통합 테스트에서 처음
재현됐다** — 합성 fixture가 2-class라 `liblinear`가 실패하지 않았기 때문이다.

- **`multi_class="multinomial"` 인자 제거**: 설계서 4절은 `LogisticRegression(multi_class=
  "multinomial")`을 고정 지정하도록 했으나, scikit-learn 1.5+에서 `multi_class` 인자
  자체가 삭제되어 `TypeError`가 발생했다. 현재 버전은 `lbfgs`류 solver가 자동으로
  multinomial로 학습하고 `liblinear`는 애초에 solver 제약상 항상 One-vs-Rest로만
  동작한다(과거 버전에서도 `solver="liblinear"` + `multi_class="multinomial"` 조합은
  허용된 적이 없다). `LogisticRegressionClassifier.__init__`/`search_hyperparameters`
  양쪽에서 `multi_class` 인자를 제거했다(`training/trainer.py`).
- **하이퍼파라미터 tie-break 정렬의 NaN 처리 버그**: `config/hyperparams_default.json`
  기본값에 있던 `solver="liblinear"`는 실제 5-class 데이터(`n_classes>=3`)에서
  scikit-learn이 fit 자체를 거부해 `NaN` 점수로 기록되는데, 최초 구현은
  `trials.sort(key=lambda t: (t.f1_macro, t.accuracy), reverse=True)`로 정렬해 NaN이
  섞이면 파이썬 정렬 비교가 깨져 **최악의 조합이 최적으로 잘못 선정**되는 버그가 있었다
  (실데이터로 재현: `C=0.01`이 최적으로 잘못 뽑힘, 실제 최적은 `C=10.0`). NaN을
  `-inf`로 치환하는 정렬 키로 교체하고, 모든 조합이 NaN이면 `DataValidationError`를
  내도록 방어 코드를 추가했다(`test_ignores_nan_scores_when_selecting_best`,
  `test_raises_when_every_combination_fails`로 기능 테스트에도 회귀 방지 케이스 추가).
  아울러 `config/hyperparams_default.json` 기본값에서 `liblinear`를 제거했다(사용자
  확인, 2026-08-20) — 5-class 문제에서 항상 실패하는 조합을 기본값에 둘 이유가 없고,
  코드 수정 없이 이 JSON만 바꾸면 되므로 [[CLAUDE.md]] 1절 OCP 원칙과도 부합한다.

## 7. 관련 문서/코드

- 요구사항/설계: [[P3_요구사항정의서_Training]], [[P3_설계서_Training]],
  [[Architecture_Design]] 2절/4절/5절
- 코드: `src/embedding_lr/training/{trainer,persistence}.py`,
  `src/embedding_lr/cli/run_phase3.py`, `src/embedding_lr/domain/models.py`(갱신),
  `config/hyperparams_default.json`
- 테스트: `tests/unit/test_domain_models.py`(갱신), `tests/integration/test_{trainer,
  persistence,run_phase3}.py`
- 실행 이미지: `docker/Dockerfile.pipeline`(`config/` 디렉터리 COPY 추가)
