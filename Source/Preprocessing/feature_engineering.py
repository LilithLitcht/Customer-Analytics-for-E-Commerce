# Chứa các hàm tạo cột mới dùng chung cho từ 2 pipeline trở lên (MBA, RFM/Clustering, Churn)

import pandas as pd

# 1. Hàm tính TotalPrice = Quantity × Price
def add_total_price_column(df: pd.DataFrame,) -> pd.DataFrame:
    required_columns = {"Quantity", "Price"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(
            f"Thiếu cột bắt buộc để tính TotalPrice: "
            f"{sorted(missing_columns)}"
        )

    result_df = df.copy()

    result_df["TotalPrice"] = (
        result_df["Quantity"] * result_df["Price"]
    )

    return result_df


# Hàm trích xuất đặc trưng thời gian
def add_datetime_features(df: pd.DataFrame, date_column: str = "InvoiceDate",) -> pd.DataFrame:

    if date_column not in df.columns:
        raise KeyError(
            f"Không tìm thấy cột ngày tháng '{date_column}'."
        )

    result_df = df.copy()

    # Đảm bảo cột ngày tháng đúng kiểu datetime
    result_df[date_column] = pd.to_datetime(
        result_df[date_column]
    )

    result_df["Year"] = result_df[date_column].dt.year
    result_df["Month"] = result_df[date_column].dt.month
    result_df["DayOfWeek"] = (result_df[date_column].dt.dayofweek)  # 0 = Thứ Hai
    result_df["Hour"] = result_df[date_column].dt.hour

    return result_df