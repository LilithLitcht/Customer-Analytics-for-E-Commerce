# Chứa các hàm phân cụm khách hàng dựa trên chỉ số RFM

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler

# Tên 3 cột RFM chuẩn - thống nhất với file customer_rfm.csv do Thành viên A xuất
RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]

# Các cột cần log-transform trước khi chuẩn hoá.
LOG_TRANSFORM_COLUMNS = ["Frequency", "Monetary"]


# Hàm chuẩn bị dữ liệu RFm trước khi phân cụm
def prepare_rfm_features(rfm_df: pd.DataFrame,
                          log_columns: list = None,
                          feature_columns: list = None) -> tuple[np.ndarray, pd.DataFrame]:
    
    if log_columns is None:
        log_columns = LOG_TRANSFORM_COLUMNS
    if feature_columns is None:
        feature_columns = RFM_COLUMNS

    transformed_df = rfm_df[feature_columns].copy()

    # log1p-transform các cột lệch mạnh
    for column in log_columns:
        if column in transformed_df.columns:
            # Dùng clip(lower=0) để tránh log(0) hoặc log(âm) gây lỗi
            transformed_df[column] = np.log1p(transformed_df[column].clip(lower=0))

    print(f"[Prepare RFM Features] Đã log-transform các cột: {log_columns}")

    # Chuẩn hoá Z-score về trung bình 0, độ lệch chuẩn 1
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(transformed_df)

    print(f"[Prepare RFM Features] Đã chuẩn hoá {scaled_features.shape[0]:,} khách hàng "
          f"trên {scaled_features.shape[1]} đặc trưng: {feature_columns}")

    return scaled_features, transformed_df


# Hàm chạy K-Means trên dữ liệu RFM đã chuẩn hoá, trả về nhãn cụm và đối tượng model
def run_kmeans(scaled_features: np.ndarray,
                n_clusters: int,
                random_state: int = 42) -> tuple[np.ndarray, KMeans]:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(scaled_features)

    print(f"[Run K-Means] k = {n_clusters}: inertia = {model.inertia_:.2f}, "
          f"kích thước các cụm = {np.bincount(labels).tolist()}")

    return labels, model


# Hàm tính Elbow và Silhouette score cho nhiều giá trị k, phục vụ chọn số cụm tối ưu.
def compute_elbow_scores(scaled_features: np.ndarray,
                          k_range: range = range(2, 11),
                          random_state: int = 42) -> pd.DataFrame:
    records = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(scaled_features)
        records.append({"k": k, "inertia": model.inertia_})
        print(f"[Compute Elbow Scores] k = {k}: inertia = {model.inertia_:.2f}")

    return pd.DataFrame(records)


# hàm tính Silhouette score cho nhiều giá trị k, phục vụ chọn số cụm tối ưu
def compute_silhouette_scores(scaled_features: np.ndarray,
                               k_range: range = range(2, 11),
                               random_state: int = 42,
                               sample_size: int = None) -> pd.DataFrame:
    records = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels,
                                  sample_size=sample_size,
                                  random_state=random_state)
        records.append({"k": k, "silhouette": score})
        print(f"[Compute Silhouette Scores] k = {k}: silhouette = {score:.4f}")

    return pd.DataFrame(records)


# Hàm tự cài đặt K-Medoids theo thuật toán PAM (Partitioning Around Medoids)
def _pam_build_phase(distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    # Điểm có tổng khoảng cách tới tất cả điểm khác nhỏ nhất
    medoid_indices = [int(np.argmin(distance_matrix.sum(axis=1)))]

    # Chọn tiếp các medoid còn lại theo chiến lược tham lam
    for _ in range(n_clusters - 1):
        # Với cấu hình medoid hiện tại, mỗi điểm đang cách medoid gần nhất bao xa
        distance_to_nearest = distance_matrix[:, medoid_indices].min(axis=1)

        gains = np.maximum(distance_to_nearest[:, None] - distance_matrix, 0).sum(axis=0)

        # Không chọn lại điểm đã là medoid
        gains[medoid_indices] = -1.0

        medoid_indices.append(int(np.argmax(gains)))

    return np.array(medoid_indices)


# Hàm chạy K-Medoids theo PAM, trả về nhãn cụm, chỉ số medoid và tổng chi phí
def run_kmedoids_pam(scaled_features: np.ndarray,
                      n_clusters: int,
                      max_iter: int = 50,
                      batch_size: int = 512,
                      metric: str = "euclidean") -> tuple[np.ndarray, np.ndarray, float]:

    n_samples = scaled_features.shape[0]

    # Tính trước ma trận khoảng cách giữa mọi cặp điểm
    distance_matrix = pairwise_distances(scaled_features, metric=metric)
    print(f"[Run K-Medoids PAM] Đã tính ma trận khoảng cách {distance_matrix.shape} "
          f"(~{distance_matrix.nbytes / 1024**2:.0f} MB RAM).")

    # PHA BUILD: Khởi tạo k medoid ban đầu
    medoid_indices = _pam_build_phase(distance_matrix, n_clusters)

    # PHA SWAP: Lặp cải thiện 
    n_iterations_done = 0
    for iteration in range(max_iter):
        n_iterations_done = iteration + 1

        # Khoảng cách từ mọi điểm tới từng medoid hiện tại
        distances_to_medoids = distance_matrix[:, medoid_indices]

        # Với mỗi điểm tìm medoid gần nhất (d1) và gần nhì (d2).
        sorted_order = np.argsort(distances_to_medoids, axis=1)
        nearest_medoid_position = sorted_order[:, 0]
        distance_to_nearest = distances_to_medoids[np.arange(n_samples), nearest_medoid_position]

        if n_clusters > 1:
            distance_to_second = distances_to_medoids[np.arange(n_samples), sorted_order[:, 1]]
        else:
            distance_to_second = np.full(n_samples, np.inf)

        current_cost = distance_to_nearest.sum()

        # Tìm phép hoán đổi tốt nhất trong vòng lặp này
        best_improvement = 0.0
        best_medoid_position = None
        best_candidate_index = None

        candidate_indices = np.setdiff1d(np.arange(n_samples), medoid_indices)

        for medoid_position in range(n_clusters):
            # Nếu gỡ medoid tại vị trí này ra, mỗi điểm sẽ cách medoid gần nhất còn lại bao xa:
            #   - Điểm đang thuộc về medoid bị gỡ thì phải dùng d2
            #   - Điểm thuộc medoid khác thì vẫn dùng d1
            baseline_distance = np.where(
                nearest_medoid_position == medoid_position,
                distance_to_second,
                distance_to_nearest,
            )

            # Đánh giá các ứng viên theo từng lô để kiểm soát bộ nhớ 
            for start in range(0, len(candidate_indices), batch_size):
                candidate_batch = candidate_indices[start:start + batch_size]

                # Chi phí mới nếu thay medoid hiện tại bằng từng ứng viên
                new_costs = np.minimum(
                    distance_matrix[:, candidate_batch],
                    baseline_distance[:, None],
                ).sum(axis=0)

                best_in_batch = int(np.argmin(new_costs))
                improvement = new_costs[best_in_batch] - current_cost

                # Dùng ngưỡng nhỏ 1e-12 để tránh hoán đổi vô nghĩa do sai số dấu phẩy động khi 2 cấu hình thực chất tương đương nhau.
                if improvement < best_improvement - 1e-12:
                    best_improvement = improvement
                    best_medoid_position = medoid_position
                    best_candidate_index = int(candidate_batch[best_in_batch])

        # Không tìm được phép hoán đổi nào cải thiện, thuật toán đã hội tụ
        if best_medoid_position is None:
            break

        medoid_indices[best_medoid_position] = best_candidate_index

    # Gán nhãn cuối cùng và tính tổng chi phí
    labels = np.argmin(distance_matrix[:, medoid_indices], axis=1)
    total_cost = distance_matrix[np.arange(n_samples), medoid_indices[labels]].sum()

    print(f"[Run K-Medoids PAM] k = {n_clusters}: hội tụ sau {n_iterations_done} vòng lặp, "
          f"tổng chi phí = {total_cost:.2f}, kích thước các cụm = {np.bincount(labels).tolist()}")

    return labels, medoid_indices, total_cost


# Hàm tổng hợp đặc điểm trung bình của từng cụm (Recency/Frequency/Monetary trung bình, số lượng khách hàng)
def summarize_clusters(rfm_df: pd.DataFrame,
                        labels: np.ndarray,
                        cluster_column: str = "Cluster") -> pd.DataFrame:

    working_df = rfm_df.copy()
    working_df[cluster_column] = labels

    summary = working_df.groupby(cluster_column).agg(
        SoKhachHang=("Recency", "size"),
        Recency_TB=("Recency", "mean"),
        Frequency_TB=("Frequency", "mean"),
        Monetary_TB=("Monetary", "mean"),
    ).round(2)

    summary["TyLePhanTram"] = (summary["SoKhachHang"] / len(working_df) * 100).round(2)

    # Sắp xếp lại thứ tự cột cho dễ đọc
    summary = summary[["SoKhachHang", "TyLePhanTram",
                       "Recency_TB", "Frequency_TB", "Monetary_TB"]]

    return summary.reset_index()


# Hàm tự động gán tên nghiệp vụ cho từng cụm dựa trên đặc điểm RFM trung bình, đảm bảo không trùng tên
def assign_segment_names(cluster_summary: pd.DataFrame,
                          cluster_column: str = "Cluster") -> dict:

    ranking_df = cluster_summary.copy()

    # Xếp hạng trên từng chiều (điểm càng cao = khách càng giá trị)
    # ascending=False cho Recency vì Recency thấp mới là tốt
    ranking_df["rank_recency"] = ranking_df["Recency_TB"].rank(ascending=False)
    ranking_df["rank_frequency"] = ranking_df["Frequency_TB"].rank(ascending=True)
    ranking_df["rank_monetary"] = ranking_df["Monetary_TB"].rank(ascending=True)

    # Điểm RFM tổng hợp
    ranking_df["rfm_score"] = (ranking_df["rank_recency"] + ranking_df["rank_frequency"] + ranking_df["rank_monetary"])

    segment_names = {}
    remaining = ranking_df.copy()

    # Cụm có điểm tổng cao nhất là Khách VIP
    vip_row = remaining.loc[remaining["rfm_score"].idxmax()]
    vip_cluster = int(vip_row[cluster_column])
    segment_names[vip_cluster] = "Khách VIP"
    remaining = remaining[remaining[cluster_column] != vip_cluster]

    # Trong các cụm còn lại, cụm lâu nhất chưa quay lại là nguy cơ rời bỏ
    if not remaining.empty:
        churn_row = remaining.loc[remaining["Recency_TB"].idxmax()]
        churn_cluster = int(churn_row[cluster_column])
        segment_names[churn_cluster] = "Khách hàng có nguy cơ rời bỏ"
        remaining = remaining[remaining[cluster_column] != churn_cluster]

    # Cụm còn lại có Frequency cao nhất là khách trung thành
    if not remaining.empty:
        loyal_row = remaining.loc[remaining["Frequency_TB"].idxmax()]
        loyal_cluster = int(loyal_row[cluster_column])
        segment_names[loyal_cluster] = "Khách hàng trung thành"
        remaining = remaining[remaining[cluster_column] != loyal_cluster]

    # Cụm còn lại có Recency thấp nhất (mua gần đây) là khách mới
    if not remaining.empty:
        new_row = remaining.loc[remaining["Recency_TB"].idxmin()]
        new_cluster = int(new_row[cluster_column])
        segment_names[new_cluster] = "Khách hàng mới"
        remaining = remaining[remaining[cluster_column] != new_cluster]

    # Các cụm dư ra (khi k > 4)
    for order, (_, row) in enumerate(remaining.iterrows(), start=1):
        segment_names[int(row[cluster_column])] = f"Khách hàng ít tương tác {order}"

    return segment_names