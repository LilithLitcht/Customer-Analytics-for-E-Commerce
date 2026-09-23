"""
==============================================================================
MODULE: evaluation.py
TÁC DỤNG: Chứa các hàm đánh giá mô hình phân lớp churn - tính Precision,
          Recall, F1, AUC, vẽ Confusion Matrix và đường cong ROC cho nhiều
          mô hình để so sánh trực quan.

NGƯỜI PHỤ TRÁCH: Thành viên D - Classification Lead (Churn)

==============================================================================
VÌ SAO KHÔNG DÙNG ACCURACY LÀM THƯỚC ĐO CHÍNH
==============================================================================
Rubric §7 tiêu chí 4 nêu rõ: phải dùng thước đo PHÙ HỢP với bài toán, và xếp
việc "chỉ báo cáo một chỉ số duy nhất không phù hợp" vào mức Đạt thấp nhất.

Với bài toán churn, Accuracy gây hiểu lầm nghiêm trọng vì hai lớp không cân
bằng. Ví dụ cụ thể từ chính dữ liệu của nhóm: nếu tỷ lệ churn thực tế là 69%,
thì một mô hình "ngớ ngẩn" chỉ cần đoán TẤT CẢ khách hàng đều churn đã đạt
Accuracy 69% - nghe có vẻ khá, nhưng mô hình đó hoàn toàn vô dụng vì không
phân biệt được ai sắp rời bỏ với ai vẫn trung thành.

Vì vậy module này luôn tính đủ bộ Precision / Recall / F1 / AUC, và hàm
evaluate_model() còn tự động tính thêm ACCURACY CỦA MÔ HÌNH NGÂY THƠ (baseline
đoán toàn bộ theo lớp đa số) để làm mốc đối chiếu - giúp trả lời ngay câu hỏi
phản biện "mô hình của bạn có thực sự tốt hơn việc đoán bừa không?".

Ý NGHĨA NGHIỆP VỤ CỦA TỪNG THƯỚC ĐO (quy ước lớp dương = 1 = churn):
  - Precision: trong số khách bị mô hình cảnh báo "sắp rời bỏ", bao nhiêu %
    thực sự rời bỏ? Precision thấp -> lãng phí ngân sách khuyến mãi cho những
    khách vốn dĩ vẫn ở lại.
  - Recall: trong số khách thực sự rời bỏ, mô hình bắt được bao nhiêu %?
    Recall thấp -> bỏ lọt khách sắp mất mà không kịp giữ chân.
  - F1: trung bình điều hoà của hai chỉ số trên, dùng khi cần cân bằng cả hai.
  - AUC: khả năng xếp hạng đúng của mô hình, không phụ thuộc vào ngưỡng cắt
    0.5 - thước đo tổng quát nhất để so sánh các mô hình với nhau.

QUY ƯỚC ĐẶT TÊN: tên hàm/biến/tham số dùng TIẾNG ANH chuẩn PEP8, comment và
docstring dùng TIẾNG VIỆT - đồng bộ với các file khác trong dự án.
==============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ==============================================================================
# 1. HÀM TÍNH BỘ CHỈ SỐ ĐÁNH GIÁ CHO MỘT MÔ HÌNH
# ==============================================================================

def evaluate_model(model,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    model_name: str = "Model") -> dict:
    """
    Tính đầy đủ bộ chỉ số đánh giá cho một mô hình trên tập kiểm thử.

    Parameters
    ----------
    model : object
        Mô hình scikit-learn đã huấn luyện, phải có phương thức predict() và
        predict_proba().
    X_test : pandas.DataFrame
        Ma trận đặc trưng tập kiểm thử. LƯU Ý: tập này KHÔNG được áp SMOTE.
    y_test : pandas.Series
        Nhãn thật của tập kiểm thử.
    model_name : str, mặc định "Model"
        Tên mô hình, dùng để hiển thị trong bảng kết quả.

    Returns
    -------
    dict
        Dictionary chứa: model_name, precision, recall, f1, auc, accuracy,
        baseline_accuracy (độ chính xác của mô hình ngây thơ đoán theo lớp
        đa số) và improvement_over_baseline.

    Ví dụ sử dụng
    -------------
    >>> metrics = evaluate_model(random_forest, X_test, y_test, "Random Forest")
    >>> metrics["f1"]
    0.78
    """
    y_predicted = model.predict(X_test)
    # Lấy xác suất thuộc lớp dương (churn = 1) để tính AUC
    y_probabilities = model.predict_proba(X_test)[:, 1]

    # zero_division=0: nếu mô hình không dự đoán dòng nào thuộc lớp dương,
    # Precision sẽ là 0/0 - tham số này đặt kết quả về 0 thay vì báo lỗi.
    precision = precision_score(y_test, y_predicted, zero_division=0)
    recall = recall_score(y_test, y_predicted, zero_division=0)
    f1 = f1_score(y_test, y_predicted, zero_division=0)
    auc = roc_auc_score(y_test, y_probabilities)
    accuracy = accuracy_score(y_test, y_predicted)

    # Baseline ngây thơ: đoán tất cả theo lớp chiếm đa số trong tập test.
    # Đây là mốc TỐI THIỂU mà mọi mô hình có ích đều phải vượt qua.
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

    print(f"[evaluate_model] {model_name}: "
          f"Precision={precision:.4f} | Recall={recall:.4f} | "
          f"F1={f1:.4f} | AUC={auc:.4f}")
    print(f"    Accuracy={accuracy:.4f} so với baseline ngây thơ "
          f"{majority_class_ratio:.4f} "
          f"(chênh lệch {accuracy - majority_class_ratio:+.4f})")

    return metrics


def build_comparison_table(metrics_list: list) -> pd.DataFrame:
    """
    Gộp kết quả đánh giá của nhiều mô hình thành một bảng so sánh duy nhất,
    sắp xếp theo F1 giảm dần.

    Parameters
    ----------
    metrics_list : list[dict]
        Danh sách các dictionary trả về từ evaluate_model().

    Returns
    -------
    pandas.DataFrame
        Bảng so sánh, mỗi dòng là một mô hình.

    Ví dụ sử dụng
    -------------
    >>> comparison = build_comparison_table([metrics_dt, metrics_rf, metrics_nb])
    """
    comparison_df = pd.DataFrame(metrics_list)
    comparison_df = comparison_df.sort_values("f1", ascending=False).reset_index(drop=True)

    return comparison_df


# ==============================================================================
# 2. HÀM VẼ CONFUSION MATRIX
# ==============================================================================

def plot_confusion_matrix(model,
                           X_test: pd.DataFrame,
                           y_test: pd.Series,
                           model_name: str = "Model",
                           ax=None) -> np.ndarray:
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix) cho một mô hình, kèm nhãn diễn
    giải bằng tiếng Việt để người đọc báo cáo hiểu ngay ý nghĩa từng ô.

    Bốn ô của ma trận (quy ước lớp dương = churn):
      - TN (trên trái) : dự đoán ở lại, thực tế ở lại  -> đúng
      - FP (trên phải) : dự đoán rời bỏ, thực tế ở lại -> báo động giả,
                         tốn chi phí khuyến mãi không cần thiết
      - FN (dưới trái) : dự đoán ở lại, thực tế rời bỏ -> BỎ LỌT khách,
                         thường là loại lỗi tốn kém nhất về mặt nghiệp vụ
      - TP (dưới phải) : dự đoán rời bỏ, thực tế rời bỏ -> đúng

    Parameters
    ----------
    model : object
        Mô hình đã huấn luyện.
    X_test : pandas.DataFrame
        Ma trận đặc trưng tập kiểm thử.
    y_test : pandas.Series
        Nhãn thật tập kiểm thử.
    model_name : str, mặc định "Model"
        Tên mô hình, hiển thị trên tiêu đề biểu đồ.
    ax : matplotlib.axes.Axes, tuỳ chọn
        Trục để vẽ lên. Nếu None, hàm tự tạo figure mới. Truyền tham số này
        khi muốn vẽ nhiều ma trận cạnh nhau trong cùng một figure.

    Returns
    -------
    numpy.ndarray
        Ma trận nhầm lẫn dạng mảng 2x2.

    Ví dụ sử dụng
    -------------
    >>> cm = plot_confusion_matrix(random_forest, X_test, y_test, "Random Forest")
    """
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
                # Đổi màu chữ theo nền để luôn đọc được
                color="white" if matrix[row, col] > threshold else "black",
                fontsize=11, fontweight="bold",
            )

    return matrix


# ==============================================================================
# 3. HÀM VẼ ĐƯỜNG CONG ROC CHO NHIỀU MÔ HÌNH
# ==============================================================================

def plot_roc_curves(models_dict: dict,
                     X_test: pd.DataFrame,
                     y_test: pd.Series,
                     ax=None) -> None:
    """
    Vẽ đường cong ROC của NHIỀU mô hình lên cùng một hệ trục để so sánh trực
    quan - mô hình nào có đường cong nằm cao hơn (gần góc trên bên trái hơn)
    thì phân biệt hai lớp tốt hơn.

    Đường chéo nét đứt là mốc tham chiếu của "đoán ngẫu nhiên" (AUC = 0.5).
    Mọi mô hình có ích đều phải nằm rõ rệt phía trên đường này.

    Parameters
    ----------
    models_dict : dict
        Dictionary dạng {tên mô hình: mô hình đã huấn luyện}.
    X_test : pandas.DataFrame
        Ma trận đặc trưng tập kiểm thử.
    y_test : pandas.Series
        Nhãn thật tập kiểm thử.
    ax : matplotlib.axes.Axes, tuỳ chọn
        Trục để vẽ lên. Nếu None, hàm tự tạo figure mới.

    Returns
    -------
    None

    Ví dụ sử dụng
    -------------
    >>> plot_roc_curves({"Decision Tree": dt, "Random Forest": rf}, X_test, y_test)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    for model_name, model in models_dict.items():
        y_probabilities = model.predict_proba(X_test)[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_probabilities)
        auc_value = roc_auc_score(y_test, y_probabilities)

        ax.plot(false_positive_rate, true_positive_rate,
                linewidth=2, label=f"{model_name} (AUC = {auc_value:.4f})")

    # Đường tham chiếu: đoán ngẫu nhiên
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            label="Đoán ngẫu nhiên (AUC = 0.5)")

    ax.set_xlabel("Tỷ lệ dương tính giả (False Positive Rate)")
    ax.set_ylabel("Tỷ lệ dương tính thật (True Positive Rate)")
    ax.set_title("So sánh đường cong ROC giữa các mô hình")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


# ==============================================================================
# 4. HÀM TRÍCH XUẤT ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG
# ==============================================================================

def get_feature_importance(model,
                            feature_names: list,
                            model_name: str = "Model") -> pd.DataFrame:
    """
    Trích xuất độ quan trọng của từng đặc trưng từ mô hình dạng cây
    (Decision Tree, Random Forest).

    Đây là phần rất có giá trị về mặt nghiệp vụ: nó trả lời câu hỏi "yếu tố
    nào quyết định nhiều nhất đến việc khách hàng rời bỏ?", giúp doanh nghiệp
    biết nên can thiệp vào đâu chứ không chỉ biết ai sắp rời bỏ.

    Parameters
    ----------
    model : object
        Mô hình đã huấn luyện. Phải có thuộc tính feature_importances_
        (Naive Bayes KHÔNG có thuộc tính này).
    feature_names : list[str]
        Danh sách tên các đặc trưng, theo đúng thứ tự cột của X_train.
    model_name : str, mặc định "Model"
        Tên mô hình.

    Returns
    -------
    pandas.DataFrame
        Bảng gồm 2 cột: "feature" và "importance", sắp xếp giảm dần.
        Trả về DataFrame rỗng nếu mô hình không hỗ trợ.

    Ví dụ sử dụng
    -------------
    >>> importance_df = get_feature_importance(random_forest, list(X_train.columns))
    """
    if not hasattr(model, "feature_importances_"):
        print(f"[get_feature_importance] {model_name} không hỗ trợ feature_importances_ "
              f"(thường gặp với Naive Bayes) - bỏ qua.")
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df["importance"] = importance_df["importance"].round(4)

    return importance_df


# ==============================================================================
# KHỐI TEST NHANH - chỉ chạy khi gọi trực tiếp "python evaluation.py"
# ==============================================================================
if __name__ == "__main__":
    print("TEST NHANH MODULE evaluation.py (dùng dữ liệu giả lập)\n")

    import matplotlib
    # Dùng backend không cần màn hình để test chạy được trong môi trường dòng lệnh
    matplotlib.use("Agg")

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB

    X_array, y_array = make_classification(
        n_samples=600, n_features=4, n_informative=3,
        n_redundant=0, n_repeated=0,
        weights=[0.7, 0.3], random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(4)]
    X_sample = pd.DataFrame(X_array, columns=feature_names)
    y_sample = pd.Series(y_array, name="Churn")

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
    )

    tree_model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
    bayes_model = GaussianNB().fit(X_train, y_train)

    print("[Test 1] evaluate_model() - kiểm tra đủ chỉ số và giá trị hợp lệ")
    metrics_tree = evaluate_model(tree_model, X_test, y_test, "Decision Tree")
    metrics_bayes = evaluate_model(bayes_model, X_test, y_test, "Naive Bayes")

    required_keys = ["precision", "recall", "f1", "auc", "accuracy", "baseline_accuracy"]
    for key in required_keys:
        assert key in metrics_tree, f"Thiếu chỉ số '{key}' trong kết quả!"
        assert 0.0 <= metrics_tree[key] <= 1.0, f"Chỉ số '{key}' phải nằm trong [0, 1]!"
    print("✔ Bộ chỉ số đầy đủ và mọi giá trị đều nằm trong khoảng hợp lệ.\n")

    print("[Test 2] build_comparison_table() - kiểm tra sắp xếp theo F1 giảm dần")
    comparison = build_comparison_table([metrics_tree, metrics_bayes])
    print(comparison[["model_name", "precision", "recall", "f1", "auc"]].to_string(index=False))
    assert comparison["f1"].is_monotonic_decreasing, "Bảng phải sắp xếp F1 giảm dần!"
    print("✔ Bảng so sánh sắp xếp đúng.\n")

    print("[Test 3] plot_confusion_matrix() - kiểm tra ma trận 2x2 và tổng khớp")
    matrix = plot_confusion_matrix(tree_model, X_test, y_test, "Decision Tree")
    plt.close()
    assert matrix.shape == (2, 2), "Confusion matrix phải có kích thước 2x2!"
    assert matrix.sum() == len(y_test), \
        "Tổng các ô trong confusion matrix phải bằng số mẫu tập test!"
    print(f"Ma trận nhầm lẫn:\n{matrix}")
    print("✔ Confusion matrix đúng kích thước và tổng khớp số mẫu.\n")

    print("[Test 4] plot_roc_curves() - kiểm tra vẽ được nhiều mô hình")
    plot_roc_curves({"Decision Tree": tree_model, "Naive Bayes": bayes_model},
                     X_test, y_test)
    plt.close()
    print("✔ Vẽ đường cong ROC cho nhiều mô hình thành công.\n")

    print("[Test 5] get_feature_importance() - kiểm tra cả trường hợp không hỗ trợ")
    importance_tree = get_feature_importance(tree_model, feature_names, "Decision Tree")
    print(importance_tree.to_string(index=False))
    assert len(importance_tree) == len(feature_names), "Phải có đủ độ quan trọng cho mọi đặc trưng!"
    assert abs(importance_tree["importance"].sum() - 1.0) < 0.01, \
        "Tổng độ quan trọng của các đặc trưng phải xấp xỉ 1.0!"

    importance_bayes = get_feature_importance(bayes_model, feature_names, "Naive Bayes")
    assert importance_bayes.empty, "Naive Bayes phải trả về bảng rỗng (không hỗ trợ)!"
    print("✔ Trích xuất độ quan trọng đúng, xử lý an toàn trường hợp không hỗ trợ.")

    print("\n✔ TEST THÀNH CÔNG toàn bộ module evaluation.py")
