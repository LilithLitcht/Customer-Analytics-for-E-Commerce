"""
==============================================================================
MODULE: models.py
TÁC DỤNG: Chứa các hàm huấn luyện mô hình phân lớp cho bài toán dự đoán churn
          - Decision Tree, Random Forest, Naive Bayes - cùng với hàm xử lý
          mất cân bằng lớp bằng SMOTE và hàm lưu model ra file.

NGƯỜI PHỤ TRÁCH: Thành viên D - Classification Lead (Churn)

NGUYÊN TẮC THIẾT KẾ QUAN TRỌNG:
Cả 3 hàm train đều dùng CHUNG MỘT CHỮ KÝ (cùng tham số đầu vào, cùng kiểu
trả về). Nhờ vậy, notebook 05 có thể gọi lặp qua danh sách thuật toán bằng
một vòng for duy nhất, thay vì viết code riêng cho từng loại - vừa gọn, vừa
đảm bảo 3 mô hình được đối xử công bằng như nhau khi so sánh.

==============================================================================
CẢNH BÁO QUAN TRỌNG NHẤT VỀ SMOTE - ĐỌC KỸ TRƯỚC KHI DÙNG
==============================================================================
SMOTE (Synthetic Minority Over-sampling Technique) sinh thêm mẫu NHÂN TẠO cho
lớp thiểu số để cân bằng dữ liệu huấn luyện.

SMOTE CHỈ ĐƯỢC ÁP DỤNG LÊN TẬP TRAIN, TUYỆT ĐỐI KHÔNG ÁP LÊN TẬP TEST.

Lý do: tập test phải phản ánh đúng phân phối THỰC TẾ mà mô hình sẽ gặp khi
triển khai. Nếu áp SMOTE lên test, ta đang đánh giá mô hình trên dữ liệu giả
- kết quả sẽ đẹp một cách sai lệch và hoàn toàn vô nghĩa. Đây là một trong
những lỗi rò rỉ dữ liệu phổ biến nhất mà rubric §7 tiêu chí 2 cảnh báo trực
tiếp ("xử lý sai cách gây rò rỉ dữ liệu").

Trong module này, hàm apply_smote() được thiết kế để CHỈ nhận tập train, và
notebook 05 phải gọi nó sau khi đã tách train/test xong.

QUY ƯỚC ĐẶT TÊN: tên hàm/biến/tham số dùng TIẾNG ANH chuẩn PEP8, comment và
docstring dùng TIẾNG VIỆT - đồng bộ với các file khác trong dự án.
==============================================================================
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Hạt giống ngẫu nhiên dùng chung toàn module - cố định để kết quả tái lập
# được giữa các lần chạy (yêu cầu bắt buộc của rubric về khả năng tái lập).
RANDOM_STATE = 42


# ==============================================================================
# 1. HÀM XỬ LÝ MẤT CÂN BẰNG LỚP BẰNG SMOTE
# ==============================================================================

def apply_smote(X_train: pd.DataFrame,
                 y_train: pd.Series,
                 random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.Series]:
    """
    Cân bằng lớp trên TẬP HUẤN LUYỆN bằng SMOTE - sinh thêm mẫu nhân tạo cho
    lớp thiểu số bằng cách nội suy giữa các mẫu thật gần nhau.

    CHỈ GỌI HÀM NÀY VỚI TẬP TRAIN. Không bao giờ gọi với tập test (xem phần
    cảnh báo ở đầu file).

    Parameters
    ----------
    X_train : pandas.DataFrame
        Ma trận đặc trưng của tập huấn luyện.
    y_train : pandas.Series
        Nhãn của tập huấn luyện (0/1).
    random_state : int, mặc định 42
        Hạt giống ngẫu nhiên để kết quả tái lập được.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        (X_resampled, y_resampled) - tập huấn luyện đã cân bằng lớp.

    Ví dụ sử dụng
    -------------
    >>> X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    """
    distribution_before = y_train.value_counts().sort_index()

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # fit_resample trả về numpy array nếu đầu vào là DataFrame ở một số phiên
    # bản - chuyển lại về DataFrame/Series để giữ tên cột, tiện cho bước sau
    # (ví dụ khi vẽ feature importance cần biết tên đặc trưng).
    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    if not isinstance(y_resampled, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y_train.name)

    distribution_after = y_resampled.value_counts().sort_index()

    print(f"[apply_smote] Phân bố lớp TRƯỚC SMOTE: {distribution_before.to_dict()} "
          f"(tổng {len(y_train):,} mẫu)")
    print(f"[apply_smote] Phân bố lớp SAU SMOTE  : {distribution_after.to_dict()} "
          f"(tổng {len(y_resampled):,} mẫu)")
    print(f"[apply_smote] Đã sinh thêm {len(y_resampled) - len(y_train):,} mẫu nhân tạo "
          f"cho lớp thiểu số.")

    return X_resampled, y_resampled


# ==============================================================================
# 2. BA HÀM HUẤN LUYỆN - CÙNG CHỮ KÝ ĐỂ DỄ SO SÁNH CÔNG BẰNG
# ==============================================================================

def train_decision_tree(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         max_depth: int = 6,
                         min_samples_leaf: int = 20,
                         random_state: int = RANDOM_STATE) -> DecisionTreeClassifier:
    """
    Huấn luyện mô hình Cây quyết định (Decision Tree).

    ƯU ĐIỂM: dễ diễn giải nhất trong 3 mô hình - có thể vẽ ra cây và chỉ cho
    người làm nghiệp vụ thấy chính xác luật quyết định (ví dụ: "nếu Recency >
    120 ngày VÀ Frequency <= 2 thì dự đoán churn"). Không cần chuẩn hoá dữ liệu.

    NHƯỢC ĐIỂM: rất dễ quá khớp (overfitting) nếu để cây mọc tự do - đây là
    lý do phải giới hạn max_depth và min_samples_leaf (kỹ thuật tỉa cây
    trước - pre-pruning, đã học ở Bài 5).

    Parameters
    ----------
    X_train : pandas.DataFrame
        Ma trận đặc trưng tập huấn luyện.
    y_train : pandas.Series
        Nhãn tập huấn luyện.
    max_depth : int, mặc định 6
        Độ sâu tối đa của cây - giới hạn để chống quá khớp.
    min_samples_leaf : int, mặc định 20
        Số mẫu tối thiểu ở mỗi nút lá - tránh tạo lá quá nhỏ chỉ khớp nhiễu.
    random_state : int, mặc định 42
        Hạt giống ngẫu nhiên.

    Returns
    -------
    sklearn.tree.DecisionTreeClassifier
        Mô hình đã huấn luyện.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    print(f"[train_decision_tree] Đã huấn luyện xong "
          f"(max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, "
          f"số lá thực tế={model.get_n_leaves()}).")

    return model


def train_random_forest(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         n_estimators: int = 200,
                         max_depth: int = 10,
                         min_samples_leaf: int = 10,
                         random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    """
    Huấn luyện mô hình Rừng ngẫu nhiên (Random Forest) - phương pháp tập hợp
    (ensemble) theo kiểu Bagging, đã học ở Bài 5.

    ƯU ĐIỂM: thường cho độ chính xác cao nhất trong 3 mô hình, ít bị quá khớp
    hơn cây đơn nhờ lấy trung bình dự đoán của nhiều cây được huấn luyện trên
    các mẫu bootstrap khác nhau. Cung cấp sẵn chỉ số độ quan trọng đặc trưng.

    NHƯỢC ĐIỂM: khó diễn giải hơn cây đơn (không thể vẽ ra 200 cây để giải
    thích cho người làm nghiệp vụ), và tốn tài nguyên tính toán hơn nhiều.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Ma trận đặc trưng tập huấn luyện.
    y_train : pandas.Series
        Nhãn tập huấn luyện.
    n_estimators : int, mặc định 200
        Số cây trong rừng. Nhiều cây hơn thường tốt hơn nhưng chậm hơn.
    max_depth : int, mặc định 10
        Độ sâu tối đa của mỗi cây.
    min_samples_leaf : int, mặc định 10
        Số mẫu tối thiểu ở mỗi nút lá.
    random_state : int, mặc định 42
        Hạt giống ngẫu nhiên.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Mô hình đã huấn luyện.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,  # dùng toàn bộ CPU sẵn có để huấn luyện nhanh hơn
    )
    model.fit(X_train, y_train)

    print(f"[train_random_forest] Đã huấn luyện xong "
          f"({n_estimators} cây, max_depth={max_depth}).")

    return model


def train_naive_bayes(X_train: pd.DataFrame,
                       y_train: pd.Series) -> GaussianNB:
    """
    Huấn luyện mô hình Naive Bayes (biến thể Gaussian, dùng cho đặc trưng số
    liên tục) - đã học ở Bài 5.

    ƯU ĐIỂM: huấn luyện cực nhanh, cần rất ít dữ liệu để đạt kết quả chấp
    nhận được, và cho ra xác suất dự đoán có cơ sở lý thuyết rõ ràng
    (định lý Bayes).

    NHƯỢC ĐIỂM: dựa trên giả định "các đặc trưng độc lập có điều kiện với
    nhau" - giả định này gần như chắc chắn BỊ VI PHẠM trong bài toán của
    nhóm, vì Frequency và Monetary rõ ràng tương quan mạnh (khách mua nhiều
    lần thì thường cũng chi nhiều tiền). Đây là điểm cần nêu thẳng thắn khi
    phân tích kết quả, và cũng là lý do Naive Bayes thường kém hơn 2 mô hình
    còn lại trên bài toán này.

    Hàm không có tham số điều chỉnh vì GaussianNB gần như không có siêu tham
    số cần tinh chỉnh - đây cũng là một đặc điểm đáng nói của mô hình này.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Ma trận đặc trưng tập huấn luyện.
    y_train : pandas.Series
        Nhãn tập huấn luyện.

    Returns
    -------
    sklearn.naive_bayes.GaussianNB
        Mô hình đã huấn luyện.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)

    print("[train_naive_bayes] Đã huấn luyện xong (GaussianNB).")

    return model


# ==============================================================================
# 3. HÀM LƯU MODEL RA FILE
# ==============================================================================

def save_model(model, output_path: str | Path) -> None:
    """
    Lưu mô hình đã huấn luyện ra file .pkl bằng joblib, phục vụ demo mà không
    phải huấn luyện lại từ đầu mỗi lần.

    Parameters
    ----------
    model : object
        Mô hình scikit-learn đã huấn luyện.
    output_path : str hoặc Path
        Đường dẫn file .pkl đích. Thư mục cha sẽ được tạo tự động nếu chưa có.

    Returns
    -------
    None

    Ví dụ sử dụng
    -------------
    >>> save_model(best_model, MODELS_DIR / "random_forest_churn.pkl")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)
    print(f"[save_model] Đã lưu mô hình: {output_path}")


# ==============================================================================
# KHỐI TEST NHANH - chỉ chạy khi gọi trực tiếp "python models.py"
# ==============================================================================
if __name__ == "__main__":
    print("TEST NHANH MODULE models.py (dùng dữ liệu giả lập)\n")

    from sklearn.datasets import make_classification

    # Tạo bài toán phân lớp nhị phân MẤT CÂN BẰNG (90% lớp 0, 10% lớp 1)
    # để kiểm tra SMOTE có thực sự cân bằng lại được hay không.
    X_array, y_array = make_classification(
        n_samples=500, n_features=5, n_informative=3,
        weights=[0.9, 0.1], random_state=RANDOM_STATE,
    )
    X_sample = pd.DataFrame(X_array, columns=[f"feature_{i}" for i in range(5)])
    y_sample = pd.Series(y_array, name="Churn")

    print(f"Dữ liệu giả lập: {len(X_sample)} mẫu, "
          f"phân bố lớp = {y_sample.value_counts().to_dict()}\n")

    print("[Test 1] apply_smote() - kiểm tra 2 lớp được cân bằng")
    X_balanced, y_balanced = apply_smote(X_sample, y_sample)
    class_counts = y_balanced.value_counts()
    assert class_counts[0] == class_counts[1], \
        "Sau SMOTE, hai lớp phải có số lượng BẰNG NHAU!"
    assert list(X_balanced.columns) == list(X_sample.columns), \
        "SMOTE phải giữ nguyên tên các cột đặc trưng!"
    print("✔ SMOTE cân bằng đúng 2 lớp và giữ nguyên tên cột.\n")

    print("[Test 2] Ba hàm train - kiểm tra cùng chữ ký, cùng kiểu trả về")
    decision_tree = train_decision_tree(X_balanced, y_balanced)
    random_forest = train_random_forest(X_balanced, y_balanced, n_estimators=50)
    naive_bayes = train_naive_bayes(X_balanced, y_balanced)

    for model_name, model in [("Decision Tree", decision_tree),
                               ("Random Forest", random_forest),
                               ("Naive Bayes", naive_bayes)]:
        predictions = model.predict(X_sample)
        probabilities = model.predict_proba(X_sample)
        assert len(predictions) == len(X_sample), f"{model_name}: số dự đoán không khớp!"
        assert probabilities.shape[1] == 2, f"{model_name}: phải trả về xác suất cho 2 lớp!"
    print("✔ Cả 3 mô hình đều huấn luyện được và trả về đúng định dạng dự đoán.\n")

    print("[Test 3] save_model() - kiểm tra lưu và load lại được")
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.pkl"
        save_model(decision_tree, model_path)
        loaded_model = joblib.load(model_path)
        assert np.array_equal(loaded_model.predict(X_sample), decision_tree.predict(X_sample)), \
            "Mô hình load lại phải cho dự đoán giống hệt mô hình gốc!"
    print("✔ Lưu và load lại mô hình hoạt động đúng.")

    print("\n✔ TEST THÀNH CÔNG toàn bộ module models.py")
