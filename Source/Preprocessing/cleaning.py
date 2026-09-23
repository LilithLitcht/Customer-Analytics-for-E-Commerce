# Chứa các hàm làm sạch dữ liệu giao dịch
import numpy as np
import pandas as pd


# Danh sách StockCode không phải sản phẩm thật (phí, chiết khấu, nghiệp vụ kế toán, dữ liệu test nội bộ...)
NON_PRODUCT_STOCK_CODES = {
    "POST":          "Phí bưu điện (Postage)",
    "DOT":           "Phí bưu điện qua kênh Dotcom (Dotcom Postage)",
    "M":             "Bút toán thủ công (Manual)",
    "D":             "Chiết khấu (Discount)",
    "BANK CHARGES":  "Phí ngân hàng (Bank Charges)",
    "CRUK":          "Khoản từ thiện Cancer Research UK (CRUK Commission)",
    "AMAZONFEE":     "Phí sàn Amazon (Amazon Fee)",
    "S":             "Mẫu thử (Samples)",
    "C2":            "Phí vận chuyển (Carriage)",
    "PADS":          "Phụ kiện đệm lót đi kèm (Pads to match cushions)",
    "ADJUST":        "Bút toán điều chỉnh sổ sách nội bộ (Adjustment)",
    "TEST001":       "Dữ liệu Test nội bộ, không phải sản phẩm thật",
    "TEST002":       "Dữ liệu Test nội bộ, không phải sản phẩm thật",
}

# Các từ khoá trong cột Description cho biết đây là bút toán điều chỉnh kho (hàng thất lạc/hư hỏng), không phải giao dịch bán hàng thật.
STOCK_ADJUSTMENT_KEYWORDS = [
    "LOST", "DAMAGE", "DAMAGES", "FOUND", "MISSING", "THROWN AWAY",
    "WRONG", "SMASHED", "CRUSHED", "MOULDY", "RUSTY", "CHECK",
]


# Hàm loại bỏ hóa đơn hủy
def remove_cancelled_invoices(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()

    invoice_as_str = result_df["Invoice"].astype(str)
    keep_mask = ~invoice_as_str.str.startswith("C")

    n_removed = (~keep_mask).sum()
    print(f"[Remove Cancelled Invoices] Đã loại {n_removed:,} dòng Invoice huỷ "
          f"(Còn lại {keep_mask.sum():,} dòng).")

    return result_df[keep_mask].reset_index(drop=True)


# Hàm loại bỏ các dòng thuộc bút toán "Adjust bad debt" (xoá nợ xấu kế toán)
def remove_bad_debt_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()

    invoice_as_str = result_df["Invoice"].astype(str)
    keep_mask = ~invoice_as_str.str.startswith("A")

    n_removed = (~keep_mask).sum()
    print(f"[Remove bad debt Adjustments] Đã loại {n_removed:,} dòng "
          f"'Adjust bad debt' (Còn lại {keep_mask.sum():,} dòng).")

    return result_df[keep_mask].reset_index(drop=True)


# Hàm loại bỏ các dòng "Quantity âm nhưng không thuộc invoice huỷ"
def remove_stock_adjustment_rows(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()

    keep_mask = result_df["Quantity"] >= 0

    n_removed = (~keep_mask).sum()
    print(f"[Remove Stock Adjustment_rows] Đã loại {n_removed:,} dòng "
          f"Quantity âm bất thường (Còn lại {keep_mask.sum():,} dòng).")

    return result_df[keep_mask].reset_index(drop=True)


# Hàm loại bỏ các dòng có StockCode KHÔNG phải sản phẩm thật (phí, chiết khấu, nghiệp vụ kế toán, dữ liệu test...)
def remove_non_product_stock_codes(df: pd.DataFrame, stock_codes: dict = None) -> pd.DataFrame:
    if stock_codes is None:
        stock_codes = NON_PRODUCT_STOCK_CODES

    result_df = df.copy()

    stock_code_upper = result_df["StockCode"].astype(str).str.upper()
    keep_mask = ~stock_code_upper.isin(stock_codes.keys())

    n_removed = (~keep_mask).sum()
    print(f"[Remove non Product Stock Codes] Đã loại {n_removed:,} dòng "
          f"StockCode phi sản phẩm (Còn lại {keep_mask.sum():,} dòng).")

    return result_df[keep_mask].reset_index(drop=True)


# Hàm loại bỏ các dòng trùng lặp hoàn toàn, chỉ giữ lại 1 bản ghi duy nhất cho mỗi nhóm trùng lặp.
def remove_duplicate_rows(df: pd.DataFrame, ignore_columns: list = None) -> pd.DataFrame:
    result_df = df.copy()

    if ignore_columns is None:
        ignore_columns = ["Sheet"] if "Sheet" in result_df.columns else []

    comparison_columns = [c for c in result_df.columns if c not in ignore_columns]

    n_before = len(result_df)
    result_df = result_df.drop_duplicates(subset=comparison_columns).reset_index(drop=True)
    n_removed = n_before - len(result_df)

    print(f"[Remove Duplicate Rows] Đã loại {n_removed:,} dòng trùng lặp "
          f"(Còn lại {len(result_df):,} dòng).")

    return result_df


# Hàm xử lý Outliers
def remove_outliers(df: pd.DataFrame,
                     column: str = "Quantity",
                     method: str = "iqr",
                     iqr_multiplier: float = 1.5,
                     zscore_threshold: float = 3.0,
                     log_transform_before_zscore: bool = True) -> pd.DataFrame:

    result_df = df.copy()

    if method == "iqr":
        q1, q3 = result_df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        keep_mask = (result_df[column] >= lower_bound) & (result_df[column] <= upper_bound)

    elif method == "zscore":
        if log_transform_before_zscore:
            values_for_zscore = np.log1p(result_df[column].clip(lower=0))
        else:
            values_for_zscore = result_df[column]

        mean_value, std_value = values_for_zscore.mean(), values_for_zscore.std()
        z_scores = (values_for_zscore - mean_value) / std_value
        keep_mask = z_scores.abs() <= zscore_threshold

    else:
        raise ValueError(
            f"method phải là 'IQR' hoặc 'Z-Score', nhận được: '{method}'"
        )

    n_removed = (~keep_mask).sum()
    print(f"[Remove Outliers] Cột '{column}', phương pháp '{method}': "
          f"đã loại {n_removed:,} dòng Outlier "
          f"(Còn lại {keep_mask.sum():,} dòng).")

    return result_df[keep_mask].reset_index(drop=True)


# Hàm Pipeline tổng hợp 
def clean_transactions(df: pd.DataFrame,
                        remove_quantity_outliers: bool = False,
                        outlier_method: str = "iqr") -> pd.DataFrame:
    print("=" * 70)
    print("BẮT ĐẦU PIPELINE LÀM SẠCH DỮ LIỆU")
    print("=" * 70)
    print(f"Số dòng ban đầu: {len(df):,}")
    print("-" * 70)

    result_df = remove_cancelled_invoices(df)
    result_df = remove_bad_debt_adjustments(result_df)
    result_df = remove_stock_adjustment_rows(result_df)
    result_df = remove_non_product_stock_codes(result_df)
    result_df = remove_duplicate_rows(result_df)

    if remove_quantity_outliers:
        result_df = remove_outliers(result_df, column="Quantity", method=outlier_method)

    print("-" * 70)
    print(f"Số dòng sau khi làm sạch: {len(result_df):,} "
          f"(đã loại tổng cộng {len(df) - len(result_df):,} dòng, "
          f"tương đương {(len(df) - len(result_df)) / len(df) * 100:.2f}%)")
    print("=" * 70)

    return result_df