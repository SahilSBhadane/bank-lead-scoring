import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score, confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

from scripts.preprocessing import build_preprocessor


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "AUC": roc_auc_score(y_test, probs),
        "PR-AUC": average_precision_score(y_test, probs),  # more honest than AUC given 11% positive rate
        "F1": f1_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "Accuracy": accuracy_score(y_test, preds),
        "Confusion Matrix": confusion_matrix(y_test, preds).tolist(),
        "Classification Report": classification_report(y_test, preds),
    }


def train_baseline(X_train, y_train, X_test, y_test):
    """Dummy classifier — always predicts the majority class. Anything we
    ship must clearly beat this, or the model isn't adding value."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    # DummyClassifier has no real predict_proba signal; use stratified for a fair AUC baseline
    dummy_stratified = DummyClassifier(strategy="stratified", random_state=42)
    dummy_stratified.fit(X_train, y_train)
    return evaluate(dummy_stratified, X_test, y_test)


def train_and_tune(X_train, y_train, X_test, y_test, include_duration=False, n_iter=15):
    preprocessor = build_preprocessor(include_duration=include_duration)

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", XGBClassifier(eval_metric="logloss", random_state=42)),
    ])

    # class imbalance: ~11% positive rate. scale_pos_weight tells XGBoost to
    # weight the minority (conversion) class higher instead of ignoring it.
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    param_dist = {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__scale_pos_weight": [1, scale_pos_weight],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=n_iter,
        scoring="average_precision",  # PR-AUC is the right target metric under class imbalance
        cv=cv, random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    metrics = evaluate(best_model, X_test, y_test)
    metrics["Best Params"] = search.best_params_
    metrics["CV Best Score (PR-AUC)"] = search.best_score_

    return best_model, metrics


def get_feature_importance(pipeline, top_n=15):
    preprocessor = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]

    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_features = list(cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]))
    all_features = list(num_features) + cat_features

    importances = clf.feature_importances_
    pairs = sorted(zip(all_features, importances), key=lambda x: x[1], reverse=True)
    return pairs[:top_n]
