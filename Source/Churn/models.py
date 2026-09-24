# Chứa các hàm huấn luyện mô hình phân lớp cho bài toán dự đoán Churn - Decision Tree, Random Forest, Naive Bayes
# Cùng với hàm xử lý mất cân bằng lớp bằng SMOTE và hàm lưu model ra file.

from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import joblib
import numpy as np
import pandas as pd


# Hạt giống ngẫu nhiên dùng chung toàn Module, cố định để kết quả tái lập được giữa các lần chạy
RANDOM_STATE = 42


# Hàm xử lý mất cân bằng lớp bằng Smote
def apply_smote(X_train: pd.DataFrame,
                 y_train: pd.Series,
                 random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.Series]:

    distribution_before = y_train.value_counts().sort_index()

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    if not isinstance(y_resampled, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y_train.name)

    distribution_after = y_resampled.value_counts().sort_index()

    print(f"[Apply Smote] Phân bố lớp trước Smote: {distribution_before.to_dict()} (Tổng {len(y_train):,} mẫu)")
    print(f"[Apply Smote] Phân bố lớp sau Smote  : {distribution_after.to_dict()} (Tổng {len(y_resampled):,} mẫu)")
    print(f"[Apply Smote] Đã sinh thêm {len(y_resampled) - len(y_train):,} mẫu nhân tạo cho lớp thiểu số")

    return X_resampled, y_resampled


# Ba hàm huấn luyện
def train_decision_tree(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         max_depth: int = 6,
                         min_samples_leaf: int = 20,
                         random_state: int = RANDOM_STATE) -> DecisionTreeClassifier:

    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state,)
    model.fit(X_train, y_train)

    print(f"[Train Decision Tree] Đã huấn luyện xong (Max Depth = {max_depth}, Min Samples Leaf = {min_samples_leaf}, Số lá thực tế = {model.get_n_leaves()}).")

    return model


def train_random_forest(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         n_estimators: int = 200,
                         max_depth: int = 10,
                         min_samples_leaf: int = 10,
                         random_state: int = RANDOM_STATE) -> RandomForestClassifier:

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=-1,)
    model.fit(X_train, y_train)

    print(f"[Train Random Forest] Đã huấn luyện xong ({n_estimators} cây, Max Depth = {max_depth}).")

    return model


def train_naive_bayes(X_train: pd.DataFrame, y_train: pd.Series) -> GaussianNB:

    model = GaussianNB()
    model.fit(X_train, y_train)

    print("[Train Naive Bayes] Đã huấn luyện xong (GaussianNB).")

    return model


# Hàm lưu Model ra file
def save_model(model, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)
    print(f"[Save Model] Đã lưu mô hình: {output_path}")