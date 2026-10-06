# Chứa các hàm huấn luyện mô hình phân lớp cho bài toán dự đoán Churn - Decision Tree, Random Forest, Naive Bayes
# Cùng với hàm xử lý mất cân bằng lớp bằng SMOTE và hàm lưu model ra file.

from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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


# Hàm tinh chỉnh siêu tham số bằng GridSearchCV
def tune_model(model_name: str,
                X_train: pd.DataFrame,
                y_train: pd.Series,
                cv_folds: int = 5,
                scoring: str = "f1",
                random_state: int = RANDOM_STATE) -> tuple:

    param_grids = {
        "decision_tree": {
            "max_depth": [4, 6, 8, 10, None],
            "min_samples_leaf": [5, 10, 20, 40],
        },
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [6, 10, 15, None],
            "min_samples_leaf": [5, 10, 20],
        },
    }

    base_estimators = {
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
    }

    if model_name not in param_grids:
        raise ValueError(f"Tên Model phải là Decision Tree hoặc Random Forest, nhận được: '{model_name}'")

    # StratifiedKFold giữ nguyên tỷ lệ lớp ở mỗi fold, quan trọng vì dữ liệu mất cân bằng
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(estimator=base_estimators[model_name], param_grid=param_grids[model_name], scoring=scoring,
                               cv=cv_strategy, n_jobs=-1, return_train_score=True,)
    grid_search.fit(X_train, y_train)

    cv_results_df = pd.DataFrame(grid_search.cv_results_).sort_values("mean_test_score", ascending=False).reset_index(drop=True)

    print(f"[Tune Model] Đã tinh chỉnh '{model_name}' bằng GridSearchCV ({cv_folds}-fold, scoring = {scoring}).")
    print(f"[Tune Model] Bộ tham số tốt nhất: {grid_search.best_params_}")
    print(f"[Tune Model] {scoring} trung bình tốt nhất (CV): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, cv_results_df

# Hàm lưu Model ra file
def save_model(model, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)
    print(f"[Save Model] Đã lưu mô hình: {output_path}")