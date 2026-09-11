"""
==============================================================================
MODULE: feature_engineering.py
TÁC DỤNG: Chứa các hàm tạo cột mới (feature) dùng CHUNG cho từ 2 pipeline
          trở lên (MBA, RFM/Clustering, Churn). Các hàm chỉ phục vụ riêng
          1 kỹ thuật (ví dụ tính nhãn churn) KHÔNG đặt ở đây - việc đó do
          B/C/D tự viết trong thư mục riêng của họ (src/mba/, src/rfm/,
          src/churn/).

NGƯỜI PHỤ TRÁCH: Thành viên A - Data & Infrastructure Lead
==============================================================================
"""

import pandas as pd


# ==============================================================================
# 1. HÀM TÍNH TotalPrice = Quantity * Price
# ==============================================================================

def tinh_total_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo cột mới "TotalPrice" = Quantity * Price cho mỗi dòng giao dịch.

    Đây là feature dùng CHUNG cho cả 3 pipeline:
    - MBA (Thành viên B): dùng để lọc/phân tích giá trị giao dịch nếu cần.
    - RFM (Thành viên A tính, Thành viên C dùng): TotalPrice là thành phần
      trực tiếp để tính chỉ số Monetary (tổng chi tiêu của khách hàng).
    - Churn (Thành viên D): TotalPrice cũng là cơ sở tính Monetary cho
      customer_churn_features.csv.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame giao dịch, cần có 2 cột "Quantity" và "Price".

    Returns
    -------
    pandas.DataFrame
        DataFrame MỚI (không sửa df gốc) có thêm cột "TotalPrice".

    Ví dụ sử dụng
    -------------
    >>> df_co_total = tinh_total_price(df_sach)
    >>> df_co_total[["Quantity", "Price", "TotalPrice"]].head()
    """
    df_moi = df.copy()
    df_moi["TotalPrice"] = df_moi["Quantity"] * df_moi["Price"]
    return df_moi


# ==============================================================================
# 2. HÀM TRÍCH XUẤT ĐẶC TRƯNG THỜI GIAN (tuỳ chọn - hữu ích cho MBA theo mùa vụ)
# ==============================================================================

def trich_xuat_dac_trung_thoi_gian(df: pd.DataFrame,
                                     cot_ngay: str = "InvoiceDate") -> pd.DataFrame:
    """
    Tách các đặc trưng thời gian từ cột ngày giao dịch: Year, Month,
    DayOfWeek, Hour - hữu ích khi cần phân tích MBA hoặc hành vi mua hàng
    theo mùa vụ / theo khung giờ (ví dụ: Thành viên B có thể dùng cột
    Month để phân tích luật kết hợp riêng cho mùa Giáng Sinh).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame giao dịch, cần có cột ngày tháng (kiểu datetime).
    cot_ngay : str, mặc định "InvoiceDate"
        Tên cột chứa thông tin ngày giờ giao dịch.

    Returns
    -------
    pandas.DataFrame
        DataFrame MỚI có thêm 4 cột: Year, Month, DayOfWeek (0=Thứ Hai,
        6=Chủ Nhật, theo chuẩn pandas), Hour.

    Ví dụ sử dụng
    -------------
    >>> df_co_thoi_gian = trich_xuat_dac_trung_thoi_gian(df_sach)
    >>> df_co_thoi_gian[["InvoiceDate", "Year", "Month", "DayOfWeek"]].head()
    """
    df_moi = df.copy()

    # Đảm bảo cột ngày tháng đúng kiểu datetime trước khi trích xuất
    df_moi[cot_ngay] = pd.to_datetime(df_moi[cot_ngay])

    df_moi["Year"] = df_moi[cot_ngay].dt.year
    df_moi["Month"] = df_moi[cot_ngay].dt.month
    df_moi["DayOfWeek"] = df_moi[cot_ngay].dt.dayofweek  # 0 = Thứ Hai
    df_moi["Hour"] = df_moi[cot_ngay].dt.hour

    return df_moi


# ==============================================================================
# KHỐI TEST NHANH - chỉ chạy khi gọi trực tiếp "python feature_engineering.py"
# ==============================================================================
if __name__ == "__main__":
    print("TEST NHANH MODULE feature_engineering.py (dùng dữ liệu giả lập)\n")

    df_test = pd.DataFrame({
        "Quantity": [6, 12, 3],
        "Price": [2.5, 1.0, 5.0],
        "InvoiceDate": pd.to_datetime([
            "2010-12-25 14:30:00",
            "2011-06-15 09:00:00",
            "2011-01-01 20:15:00",
        ]),
    })

    print("Dữ liệu giả lập ban đầu:")
    print(df_test)

    print("\n[Test 1] tinh_total_price()")
    df_total = tinh_total_price(df_test)
    print(df_total)
    assert list(df_total["TotalPrice"]) == [15.0, 12.0, 15.0], "Test TotalPrice thất bại!"
    print("✔ TotalPrice tính đúng.")

    print("\n[Test 2] trich_xuat_dac_trung_thoi_gian()")
    df_thoi_gian = trich_xuat_dac_trung_thoi_gian(df_test)
    print(df_thoi_gian[["InvoiceDate", "Year", "Month", "DayOfWeek", "Hour"]])
    assert list(df_thoi_gian["Year"]) == [2010, 2011, 2011], "Test Year thất bại!"
    print("✔ Đặc trưng thời gian tách đúng.")

    print("\n✔ TEST THÀNH CÔNG toàn bộ module feature_engineering.py")