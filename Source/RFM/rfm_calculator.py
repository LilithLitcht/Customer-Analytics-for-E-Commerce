# Chứa hàm tính Recency/Frequency/Monetary (RFM) cho từng khách hàng

import pandas as pd


def calculate_rfm(df: pd.DataFrame,
                   snapshot_date,
                   customer_id_col: str = "Customer ID",
                   invoice_col: str = "Invoice",
                   date_col: str = "InvoiceDate",
                   total_price_col: str = "TotalPrice") -> pd.DataFrame:
    working_df = df.copy()

    # Chuẩn hoá snapshot_date về đúng kiểu Timestamp để so sánh chính xác
    snapshot_date = pd.Timestamp(snapshot_date)

    # Đảm bảo cột ngày tháng đúng kiểu datetime trước khi so sánh/tính toán
    working_df[date_col] = pd.to_datetime(working_df[date_col])

    # Loại các dòng thiếu Customer ID - Không thể tính RFM cho khách hàng không xác định được danh tính.
    n_before = len(working_df)
    working_df = working_df[working_df[customer_id_col].notna()].copy()
    print(f"[Calculate RFM] Đã loại {n_before - len(working_df):,} dòng thiếu "
          f"'{customer_id_col}' (Còn lại {len(working_df):,} dòng để tính RFM).")

    # Ngăn chặn rò rỉ dữ liệu khi hàm này được gọi với snapshot_date
    n_before_date_filter = len(working_df)
    working_df = working_df[working_df[date_col] <= snapshot_date].copy()
    n_removed_after_snapshot = n_before_date_filter - len(working_df)

    print(f"[Calculate RFM] Mốc Snapshot Date = {snapshot_date.date()}. "
          f"Đã loại {n_removed_after_snapshot:,} dòng có ngày giao dịch "
          f"Sau mốc này (Còn lại {len(working_df):,} dòng).")

    if n_removed_after_snapshot == 0:
        print("  (Lưu ý: 0 dòng bị loại nghĩa là Snapshot Date đang lớn hơn "
              "hoặc bằng ngày giao dịch cuối cùng trong dữ liệu.")

    # Gộp nhóm theo Customer ID, tính Recency/Frequency/Monetary
    rfm_table = working_df.groupby(customer_id_col).agg(
        LastPurchaseDate=(date_col, "max"),
        Frequency=(invoice_col, "nunique"),
        Monetary=(total_price_col, "sum"),
    ).reset_index()

    # Recency = Số ngày từ lần mua gần nhất đến snapshot_date
    rfm_table["Recency"] = (snapshot_date - rfm_table["LastPurchaseDate"]).dt.days

    # Sắp xếp lại cột theo đúng thứ tự chuẩn đã thống nhất với nhóm
    rfm_table = rfm_table[[customer_id_col, "Recency", "Frequency", "Monetary"]]

    print(f"[Calculate RFM] Đã tính RFM cho {len(rfm_table):,} khách hàng.")

    return rfm_table