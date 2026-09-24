# Chứa các hàm đánh giá mô hình phân lớp Churn, tính Precision, Recall, F1, AUC, vẽ Confusion Matrix và đường cong ROC cho nhiều mô hình để so sánh trực quan.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,)


# Hàm tính bộ chỉ số đánh giá cho một mô hình
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model") -> dict:

    y_predicted = model.predict(X_test)
    # Lấy xác suất thuộc lớp dương (Churn = 1) để tính AUC
    y_probabilities = model.predict_proba(X_test)[:, 1]

    # Nếu mô hình không dự đoán dòng nào thuộc lớp dương, Precision sẽ là 0/0 - tham số này đặt kết quả về 0 thay vì báo lỗi.
    precision = precision_score(y_test, y_predicted, zero_division=0)
    recall = recall_score(y_test, y_predicted, zero_division=0)
    f1 = f1_score(y_test, y_predicted, zero_division=0)
    auc = roc_auc_score(y_test, y_probabilities)
    accuracy = accuracy_score(y_test, y_predicted)

    # Đoán tất cả theo lớp chiếm đa số trong tập test
    majority_class_ratio = max(y_test.mean(), 1 - y_test.mean())

    metrics = {
        "model_name": model_name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "baseline_accuracy": round(majority_class_ratio, 4),
        "improvement_over_baseline": round(accuracy - majority_class_ratio, 4),
    }

    print(f"[Evaluate Model] {model_name}: Precision = {precision:.4f}  |  Recall = {recall:.4f}  |  F1 = {f1:.4f}  |  AUC = {auc:.4f}")
    print(f"    Accuracy = {accuracy:.4f} so với Naive Baseline {majority_class_ratio:.4f} (Chênh lệch {accuracy - majority_class_ratio:+.4f})")

    return metrics


# Hàm gộp kết quả đánh giá nhiều mô hình thành 1 bảng so sánh duy nhất theo F1 giảm dần
def build_comparison_table(metrics_list: list) -> pd.DataFrame:
    comparison_df = pd.DataFrame(metrics_list)
    comparison_df = comparison_df.sort_values("f1", ascending=False).reset_index(drop=True)

    return comparison_df


# Hàm vẽ Confusion Matrix
def plot_confusion_matrix(model,X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model", ax=None) -> np.ndarray:
    y_predicted = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_predicted)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))

    image = ax.imshow(matrix, cmap="Blues")

    tick_labels = ["Ở lại (0)", "Rời bỏ (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Mô hình dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"Confusion Matrix - {model_name}")

    # Ghi số lượng và nhãn viết tắt vào từng ô
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    threshold = matrix.max() / 2
    for row in range(2):
        for col in range(2):
            ax.text(
                col, row,
                f"{cell_labels[row][col]}\n{matrix[row, col]:,}",
                ha="center", va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontsize=11, fontweight="bold",
            )

    return matrix


# Hàm vẽ đường cong ROC cho nhiều mô hình
def plot_roc_curves(models_dict: dict,X_test: pd.DataFrame, y_test: pd.Series, ax=None) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    for model_name, model in models_dict.items():
        y_probabilities = model.predict_proba(X_test)[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_probabilities)
        auc_value = roc_auc_score(y_test, y_probabilities)

        ax.plot(false_positive_rate, true_positive_rate,
                linewidth=2, label=f"{model_name} (AUC = {auc_value:.4f})")

    # Đường tham chiếu đoán ngẫu nhiên
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            label="Đoán ngẫu nhiên (AUC = 0.5)")

    ax.set_xlabel("Tỷ lệ dương tính giả (False Positive Rate)")
    ax.set_ylabel("Tỷ lệ dương tính thật (True Positive Rate)")
    ax.set_title("So sánh đường cong ROC giữa các mô hình")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


# Hàm trích xuất độ quan trọng của đặc trưng
def get_feature_importance(model,feature_names: list, model_name: str = "Model") -> pd.DataFrame:
 
    if not hasattr(model, "feature_importances_"):
        print(f"[Get Feature Importance] {model_name} không hỗ trợ Feature Importances (Thường gặp với Naive Bayes) - Bỏ qua.")
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df["importance"] = importance_df["importance"].round(4)

    return importance_df