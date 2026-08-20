"""Phase 3 학습 오케스트레이션 — P3_설계서_Training.md 4절. 등급 B(오케스트레이션) —
`Classifier` Protocol(domain/interfaces.py) 구현체 + GridSearchCV/PredefinedSplit 탐색.

Trigger: search_hyperparameters(...) / train_final_model(...) — cli/run_phase3.py가 감싼다.
Input:   train_vectors.parquet/test_vectors.parquet(load_vectors 경유), param_grid(dict)
Output:  HyperparamSearchResult(탐색 이력) / LogisticRegressionClassifier(최종 모델)
"""

import math

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from embedding_lr.constants import CLASS_LABELS
from embedding_lr.domain.models import HyperparamSearchResult, HyperparamTrial
from embedding_lr.exceptions import DataValidationError

_REQUIRED_COLUMNS = ["embedding", "label"]


class LogisticRegressionClassifier:
    """Classifier Protocol 구현체 — sklearn LogisticRegression 래핑.

    params는 GridSearchCV 탐색 대상(C/solver/max_iter)만 받는다. scikit-learn(설치 버전
    >=1.5)은 `multi_class` 인자를 제거했다 — `lbfgs`류 solver는 자동으로 multinomial로
    학습하고, `liblinear`는 solver 자체 제약상 항상 One-vs-Rest로만 동작한다
    (Scope_Definition.md 4.1절의 "multinomial 고정"은 liblinear를 탐색 범위에 둔 이상
    solver 선택에 따라 갈린다).
    """

    def __init__(self, **params) -> None:
        self._model = LogisticRegression(**params)

    def fit(self, X: list[list[float]], y: list[str]) -> None:
        self._model.fit(X, y)

    def predict_proba(self, X: list[list[float]]) -> list[dict[str, float]]:
        proba = self._model.predict_proba(X)
        labels = list(self._model.classes_)
        return [dict(zip(labels, row)) for row in proba]


def load_vectors(path: str) -> tuple[list[list[float]], list[str]]:
    """<split>_vectors.parquet을 읽어 (embedding 컬럼, label 컬럼) 반환.
    컬럼 누락 또는 label 값이 CLASS_LABELS에 없으면 DataValidationError."""
    df = pd.read_parquet(path)

    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise DataValidationError(f"{path}: 필수 컬럼 누락 {missing}")

    unknown_labels = sorted(set(df["label"]) - set(CLASS_LABELS))
    if unknown_labels:
        raise DataValidationError(f"{path}: 알 수 없는 label 값 {unknown_labels}, 허용값 {CLASS_LABELS}")

    X = [list(embedding) for embedding in df["embedding"]]
    y = list(df["label"])
    return X, y


def search_hyperparameters(
    X_train: list[list[float]],
    y_train: list[str],
    X_test: list[list[float]],
    y_test: list[str],
    param_grid: dict[str, list],
) -> HyperparamSearchResult:
    """GridSearchCV + PredefinedSplit(train=-1, test=0)으로 각 조합을 train으로 학습,
    test로 평가 — k-fold 교차검증은 일어나지 않는다. F1-macro 내림차순 → Accuracy
    내림차순으로 정렬해 1위 조합을 best로 선정한다.

    일부 조합(예: solver="liblinear" + 3-class 이상)은 sklearn이 fit 자체를 거부해
    NaN 점수로 기록될 수 있다 — NaN은 최하위로 취급해 정렬하며, 모든 조합이 NaN이면
    선정 불가로 보고 DataValidationError를 낸다."""
    X = X_train + X_test
    y = y_train + y_test
    test_fold = [-1] * len(X_train) + [0] * len(X_test)

    search = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid=param_grid,
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
        cv=PredefinedSplit(test_fold),
        refit=False,
    )
    search.fit(X, y)

    results = search.cv_results_
    trials = [
        HyperparamTrial(
            params=results["params"][i],
            accuracy=float(results["mean_test_accuracy"][i]),
            f1_macro=float(results["mean_test_f1_macro"][i]),
        )
        for i in range(len(results["params"]))
    ]
    def _sort_key(trial: HyperparamTrial) -> tuple[float, float]:
        f1 = trial.f1_macro if not math.isnan(trial.f1_macro) else float("-inf")
        accuracy = trial.accuracy if not math.isnan(trial.accuracy) else float("-inf")
        return (f1, accuracy)

    trials.sort(key=_sort_key, reverse=True)
    best = trials[0]

    if math.isnan(best.f1_macro):
        raise DataValidationError("모든 하이퍼파라미터 조합이 fit에 실패했습니다(NaN 점수) — param_grid를 확인하세요")

    return HyperparamSearchResult(
        best_params=best.params,
        best_accuracy=best.accuracy,
        best_f1_macro=best.f1_macro,
        trials=trials,
    )


def train_final_model(
    X_train: list[list[float]], y_train: list[str], best_params: dict[str, float | int | str],
) -> LogisticRegressionClassifier:
    """best_params로 새 LogisticRegressionClassifier를 만들어 train set만으로 재학습 —
    search_hyperparameters()가 내부에서 만든 추정기(train+test 결합 학습됨)를 재사용하지 않는다."""
    classifier = LogisticRegressionClassifier(**best_params)
    classifier.fit(X_train, y_train)
    return classifier
