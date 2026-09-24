# Sinh nhãn Churn cho bài toán phân lớp dự đoán khách hàng rời bỏ, dựa trên phân tích thống kê khoảng cách giữa các lần mua liên tiếp thay vì chọn ngưỡng tuỳ tiện.

import numpy as np
import pandas as pd

# Tính khoảng cách giữa các lần mua liên tiếp
def compute_inter_purchase_gaps(df: pd.DataFrame,
                                  customer_id_col: str = "Customer ID",
                                  date_col: str = "InvoiceDate") -> pd.Series:

    working_df = df[df[customer_id_col].notna()].copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col])

    # Chuẩn hoá về mốc ngày (00:00) để gộp các hoá đơn trong cùng một ngày
    working_df["PurchaseDate"] = working_df[date_col].dt.normalize()

    # Với mỗi khách lấy danh sách ngày mua duy nhất, sắp xếp tăng dần
    purchase_dates_per_customer = (
        working_df.groupby(customer_id_col)["PurchaseDate"]
        .apply(lambda dates: sorted(dates.unique()))
    )

    all_gaps = []
    for purchase_dates in purchase_dates_per_customer:
        if len(purchase_dates) > 1:
            date_series = pd.Series(pd.to_datetime(list(purchase_dates)))
            gaps_in_days = date_series.diff().dropna().dt.days
            all_gaps.extend(gaps_in_days.tolist())

    gaps_series = pd.Series(all_gaps, name="InterPurchaseGap")

    n_repeat_customers = (purchase_dates_per_customer.apply(len) > 1).sum()
    n_one_time_customers = (purchase_dates_per_customer.apply(len) == 1).sum()

    print(f"[Compute Inter Purchase Gaps] Thu được {len(gaps_series):,} khoảng cách "
          f"từ {n_repeat_customers:,} khách mua lặp lại.")
    print(f"[Compute Inter Purchase Gaps] Có {n_one_time_customers:,} khách chỉ mua đúng 1 lần "
          f"({n_one_time_customers / len(purchase_dates_per_customer) * 100:.1f}%) - ")

    return gaps_series


# Hàm lập bảng tóm tắt phân phối inter-purchase gap theo các mốc percentile
def summarize_gap_distribution(gaps: pd.Series,
                                percentiles: list = None) -> pd.DataFrame:

    if percentiles is None:
        percentiles = [50, 60, 70, 75, 80, 85, 90, 95]

    records = []
    for p in percentiles:
        gap_value = gaps.quantile(p / 100)
        records.append({
            "percentile": f"P{p}",
            "gap_days": round(gap_value, 1),
            "y_nghia": f"{p}% Số lần mua liên tiếp cách nhau không quá {gap_value:.0f} ngày",
        })

    return pd.DataFrame(records)


# Hàm đề xuất độ dài của sổ Churn (số ngày) dựa trên 1 mốc percentile của phân phối inter-purchase gap.
def suggest_churn_window(gaps: pd.Series, percentile: int = 75) -> int:
    window_days = int(gaps.quantile(percentile / 100))

    print(f"[Suggest Churn Window] Cửa sổ Churn đề xuất theo P{percentile}: {window_days} ngày.")
    print(f"  ⇒ Khách hàng không quay lại mua trong {window_days} ngày được coi là đã rời bỏ, vì {percentile}% các chu kỳ mua thực tế đều ngắn hơn khoảng này.")

    return window_days


# Sinh nhãn Churn tại 1 mốc Snapshot
def generate_churn_labels(df: pd.DataFrame,
                           snapshot_date,
                           window_days: int,
                           customer_id_col: str = "Customer ID",
                           date_col: str = "InvoiceDate") -> pd.DataFrame:

    working_df = df[df[customer_id_col].notna()].copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col])

    snapshot_timestamp = pd.Timestamp(snapshot_date)
    window_end = snapshot_timestamp + pd.Timedelta(days=window_days)

    # Kiểm tra cửa sổ quan sát có bị cụt không
    last_date_in_data = working_df[date_col].max()
    if window_end > last_date_in_data:
        n_days_short = (window_end - last_date_in_data).days
        print(f"[Generate Churn Labels] Cảnh báo: Cửa sổ quan sát kết thúc ngày {window_end.date()} nhưng dữ liệu chỉ có đến {last_date_in_data.date()} (Thiếu {n_days_short} ngày).")

    # Khách đã hoạt động tính đến thời điểm snapshot
    customers_before_snapshot = working_df[
        working_df[date_col] <= snapshot_timestamp
    ][customer_id_col].unique()

    # Khách có quay lại mua trong cửa sổ quan sát
    customers_returned = working_df[
        (working_df[date_col] > snapshot_timestamp)
        & (working_df[date_col] <= window_end)
    ][customer_id_col].unique()

    labels_df = pd.DataFrame({customer_id_col: customers_before_snapshot})
    # Churn = 1 khi khách không nằm trong danh sách quay lại mua
    labels_df["Churn"] = (~labels_df[customer_id_col].isin(customers_returned)).astype(int)

    churn_rate = labels_df["Churn"].mean()
    print(f"[Generate Churn Labels] Snapshot = {snapshot_timestamp.date()}, cửa sổ = {window_days} ngày -> {len(labels_df):,} khách hàng, Churn = {labels_df['Churn'].sum():,} ({churn_rate * 100:.1f}%), không Churn = {(labels_df['Churn'] == 0).sum():,}")

    return labels_df


# Tạo 2 mốc Snapshot để tách Train/Test theo thời gian
def build_time_based_snapshots(df: pd.DataFrame,
                                 window_days: int,
                                 date_col: str = "InvoiceDate") -> tuple[pd.Timestamp, pd.Timestamp]:

    date_series = pd.to_datetime(df[date_col])
    last_date = date_series.max().normalize()

    test_snapshot = last_date - pd.Timedelta(days=window_days)
    train_snapshot = test_snapshot - pd.Timedelta(days=window_days)

    print(f"[Build Time Based Snapshots] Ngày cuối dữ liệu: {last_date.date()}")
    print(f"  TRAIN: Snapshot = {train_snapshot.date()}, cửa sổ quan sát nhãn = ({train_snapshot.date()}, {test_snapshot.date()}]")
    print(f"  TEST : Snapshot = {test_snapshot.date()}, cửa sổ quan sát nhãn = ({test_snapshot.date()}, {last_date.date()}]")

    # Kiểm tra mốc Train có đủ dữ liệu lịch sử phía trước không
    first_date = date_series.min().normalize()
    history_days = (train_snapshot - first_date).days
    if history_days < window_days:
        print(f"  Cảnh báo: Chỉ có {history_days} ngày lịch sử trước mốc Train - Có thể không đủ để tính đặc trưng RFM đáng tin cậy.")

    return train_snapshot, test_snapshot


# Bổ sung đặc trưng phát sinh cho bài toán Churn
def add_churn_features(rfm_df: pd.DataFrame,
                        transactions_df: pd.DataFrame,
                        snapshot_date,
                        customer_id_col: str = "Customer ID",
                        date_col: str = "InvoiceDate") -> pd.DataFrame:

    snapshot_timestamp = pd.Timestamp(snapshot_date)

    working_df = transactions_df[transactions_df[customer_id_col].notna()].copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col])
    
    # Chỉ dùng dữ liệu quá khứ, là rào chắn chống rò rỉ dữ liệu
    working_df = working_df[working_df[date_col] <= snapshot_timestamp]
    working_df["PurchaseDate"] = working_df[date_col].dt.normalize()

    result_df = rfm_df.copy()

    # Giá trị trung bình mỗi hoá đơn
    # replace(0, np.nan) để tránh chia cho 0, sau đó điền lại bằng 0
    result_df["AvgOrderValue"] = (
        result_df["Monetary"] / result_df["Frequency"].replace(0, np.nan)
    ).fillna(0).round(2)

    # Số ngày từ lần mua đầu tiên đến Snapshot
    first_purchase = working_df.groupby(customer_id_col)["PurchaseDate"].min()
    tenure_days = (snapshot_timestamp - first_purchase).dt.days
    result_df["Tenure"] = result_df[customer_id_col].map(tenure_days).fillna(0).astype(int)

    # Khoảng cách trung bình giữa các lần mua của riêng từng khách
    def average_gap_of_customer(purchase_dates: pd.Series) -> float:
        unique_dates = sorted(purchase_dates.unique())
        if len(unique_dates) < 2:
            # Khách chỉ mua 1 lần: không có khoảng cách nào để tính.
            # Dùng NaN ở đây rồi điền sau, không dùng 0 vì 0 mang nghĩa
            return np.nan
        date_series = pd.Series(pd.to_datetime(list(unique_dates)))
        return date_series.diff().dropna().dt.days.mean()

    avg_gap = working_df.groupby(customer_id_col)["PurchaseDate"].apply(average_gap_of_customer)
    result_df["AvgGapDays"] = result_df[customer_id_col].map(avg_gap)

    # Khách mua 1 lần: điền bằng chính Tenure của họ 
    # lặng suốt từ lần mua đầu đến giờ", hợp lý hơn nhiều so với điền 0.
    result_df["AvgGapDays"] = result_df["AvgGapDays"].fillna(result_df["Tenure"]).round(1)

    print(f"[Add Churn Features] Đã bổ sung 3 đặc trưng: AvgOrderValue, Tenure, AvgGapDays cho {len(result_df):,} khách hàng.")

    return result_df