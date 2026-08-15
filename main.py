import os
import joblib
from sklearn.model_selection import train_test_split

from scripts.data_loader import load_data
from scripts.preprocessing import split_features_target
from scripts.modeling import train_baseline, train_and_tune, get_feature_importance


def run(include_duration: bool, label: str, X_full_train, X_full_test, y_train, y_test, feature_cols):
    X_train = X_full_train[feature_cols]
    X_test = X_full_test[feature_cols]
    print(f"\n🚀 Training [{label}] ...")
    model, metrics = train_and_tune(X_train, y_train, X_test, y_test, include_duration=include_duration)
    print(f"📊 [{label}] AUC: {metrics['AUC']:.3f} | PR-AUC: {metrics['PR-AUC']:.3f} | "
          f"F1: {metrics['F1']:.3f} | Precision: {metrics['Precision']:.3f} | Recall: {metrics['Recall']:.3f}")
    return model, metrics


def main():
    os.makedirs("outputs", exist_ok=True)

    print("🔄 Loading data...")
    df = load_data()

    print("⚙️  Splitting features/target...")
    # Build the full feature set once (with duration included) so both the
    # realistic and benchmark runs share the exact same train/test split —
    # this keeps the comparison between them fair.
    X_all, y = split_features_target(df, include_duration=True)
    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )

    realistic_cols = [c for c in X_all.columns if c != "duration"]
    benchmark_cols = list(X_all.columns)

    print("📉 Establishing dummy baseline (must beat this)...")
    baseline_metrics = train_baseline(
        X_train_all[realistic_cols], y_train, X_test_all[realistic_cols], y_test
    )
    print(f"📊 [Dummy Baseline] AUC: {baseline_metrics['AUC']:.3f} | "
          f"PR-AUC: {baseline_metrics['PR-AUC']:.3f}")

    # --- THE REAL DELIVERABLE ---
    # No 'duration' — only information available BEFORE a call happens.
    real_model, real_metrics = run(
        include_duration=False, label="REAL MODEL (no duration, pre-call features only)",
        X_full_train=X_train_all, X_full_test=X_test_all,
        y_train=y_train, y_test=y_test, feature_cols=realistic_cols,
    )

    # --- BENCHMARK-ONLY, NOT SHIPPED ---
    # Included 'duration' on purpose to show the leakage effect quantitatively,
    # exactly as UCI's own docs warn against using it for real prediction.
    _, benchmark_metrics = run(
        include_duration=True, label="BENCHMARK ONLY (with duration - leaky, for comparison)",
        X_full_train=X_train_all, X_full_test=X_test_all,
        y_train=y_train, y_test=y_test, feature_cols=benchmark_cols,
    )

    print("📊 Computing feature importance for real model...")
    top_features = get_feature_importance(real_model)

    print("💾 Saving model artifact...")
    joblib.dump(real_model, "outputs/model.joblib")

    print("📝 Writing metrics report...")
    with open("outputs/metrics_report.txt", "w") as f:
        f.write("BANK TERM DEPOSIT LEAD SCORING — MODEL REPORT\n")
        f.write("=" * 60 + "\n\n")

        for label, m in [
            ("Dummy Baseline (stratified random)", baseline_metrics),
            ("REAL MODEL — no duration (deliverable)", real_metrics),
            ("Benchmark ONLY — with duration (leaky, not shipped)", benchmark_metrics),
        ]:
            f.write(f"--- {label} ---\n")
            f.write(f"AUC:       {m['AUC']:.4f}\n")
            f.write(f"PR-AUC:    {m['PR-AUC']:.4f}\n")
            f.write(f"F1:        {m['F1']:.4f}\n")
            f.write(f"Precision: {m['Precision']:.4f}\n")
            f.write(f"Recall:    {m['Recall']:.4f}\n")
            f.write(f"Accuracy:  {m['Accuracy']:.4f}\n")
            if "Best Params" in m:
                f.write(f"Best Params: {m['Best Params']}\n")
            f.write(f"\nConfusion Matrix:\n{m['Confusion Matrix']}\n")
            f.write(f"\n{m['Classification Report']}\n")
            f.write("\n" + "=" * 60 + "\n\n")

        f.write("Top Feature Importances (real model)\n")
        f.write("-" * 40 + "\n")
        for feat, score in top_features:
            f.write(f"{feat:40s} {score:.4f}\n")

    print("\n✅ Done. Check outputs/metrics_report.txt and outputs/model.joblib")


if __name__ == "__main__":
    main()
