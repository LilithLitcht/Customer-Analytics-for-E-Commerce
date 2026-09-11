"""
==============================================================================
MODULE: rfm_calculator.py
TÁC DỤNG: Chứa hàm tính Recency/Frequency/Monetary (RFM) cho từng khách
          hàng. Đây là file ĐẶC BIỆT QUAN TRỌNG - là input chung cho CẢ
          pipeline của Thành viên C (Clustering) và Thành viên D (Churn),
          nhưng phải tính theo 2 CÁCH KHÁC NHAU cho 2 người. Đây là điểm
          dễ gây lỗi "rò rỉ dữ liệu" (data leakage) NHẤT của toàn dự án
          nếu làm sai.

NGƯỜI PHỤ TRÁCH: Thành viên A - Data & Infrastructure Lead

==============================================================================
GIẢI THÍCH CƠ CHẾ "1 HÀM NHƯNG GỌI 2 LẦN KHÁC NHAU" - ĐỌC KỸ TRƯỚC KHI DÙNG
==============================================================================

Hàm tinh_rfm() bên dưới nhận tham số snapshot_date làm "mốc thời gian tham
chiếu" để tính Recency. Cùng một hàm này sẽ được gọi 2 LẦN với 2 mục đích
khác nhau trong notebooks/02_data_preparation.ipynb:

  * GỌI LẦN 1 - cho Thành viên C (Clustering / Segmentation):
      snapshot_date = ngày cuối cùng có trong TOÀN BỘ dữ liệu (hoặc 1 ngày
      sau đó 1 chút, ví dụ +1 ngày).
      => Mục đích: MÔ TẢ khách hàng dựa trên TOÀN BỘ lịch sử mua hàng đã
      biết. Không có khái niệm "dự đoán tương lai" ở đây nên dùng toàn bộ
      dữ liệu là hợp lệ và hợp lý.
      => Kết quả lưu vào: data/processed/customer_rfm.csv

  * GỌI LẦN 2 - cho Thành viên D (Churn Prediction):
      snapshot_date = một mốc thời gian CẮT SỚM HƠN ngày cuối cùng của dữ
      liệu - do THÀNH VIÊN D tính toán và đề xuất, dựa trên phân tích phân
      phối inter-purchase gap (xem src/churn/label_generator.py).
      => Mục đích: để dành phần dữ liệu SAU mốc snapshot dùng để kiểm tra
      khách hàng có thực sự quay lại mua hay không (tức là để D sinh nhãn
      churn). Nếu dùng TOÀN BỘ dữ liệu để tính RFM rồi mới "dự đoán churn",
      mô hình sẽ "nhìn thấy trước tương lai" - đây là lỗi RÒ RỈ DỮ LIỆU
      (data leakage) nghiêm trọng, khiến kết quả đánh giá mô hình không
      còn đáng tin cậy.
      => Kết quả lưu vào: data/processed/customer_churn_features.csv

QUY TẮC BẮT BUỘC: mốc snapshot_date cụ thể dùng cho Thành viên D KHÔNG được
tự quyết định một mình bởi Thành viên A - phải trao đổi và thống nhất với D
TRƯỚC khi gọi hàm lần 2, vì D là người có căn cứ thống kê (percentile của
inter-purchase gap) để chọn con số này.
==============================================================================
"""

import pandas as pd


def tinh_rfm(df: pd.DataFrame,
             snapshot_date,
             cot_customer_id: str = "Customer ID",
             cot_invoice: str = "Invoice",
             cot_ngay: str = "InvoiceDate",
             cot_total_price: str = "TotalPrice") -> pd.DataFrame:
    """
    Tính 3 chỉ số Recency, Frequency, Monetary (RFM) cho từng khách hàng,
    dựa trên 1 mốc thời gian tham chiếu (snapshot_date) do người gọi hàm
    quyết định.

    Định nghĩa 3 chỉ số:
    - Recency (R)  : số ngày từ lần mua GẦN NHẤT của khách hàng đến
                     snapshot_date. R càng NHỎ = khách càng mới hoạt động.
    - Frequency (F): số lượng Invoice (hoá đơn) DUY NHẤT của khách hàng
                     (tính đến trước hoặc bằng snapshot_date). F càng LỚN
                     = khách càng trung thành.
    - Monetary (M) : tổng TotalPrice khách hàng đã chi tiêu (tính đến
                     trước hoặc bằng snapshot_date). M càng LỚN = khách
                     càng có giá trị.

    QUAN TRỌNG VỀ RÒ RỈ DỮ LIỆU: hàm này CHỈ tính RFM dựa trên các giao
    dịch xảy ra TRƯỚC HOẶC BẰNG snapshot_date - các giao dịch xảy ra SAU
    mốc này sẽ bị loại khỏi phép tính. Đây chính là cơ chế đảm bảo không
    rò rỉ thông tin "tương lai" khi dùng cho bài toán dự đoán churn.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame giao dịch đã làm sạch, cần có các cột: Customer ID,
        Invoice, InvoiceDate, TotalPrice. Các dòng thiếu Customer ID sẽ
        TỰ ĐỘNG bị loại trước khi tính (vì không thể tính RFM cho khách
        hàng không xác định được danh tính).
    snapshot_date : str hoặc datetime hoặc pandas.Timestamp
        Mốc thời gian tham chiếu để tính Recency. Có thể truyền vào dạng
        chuỗi (ví dụ "2011-12-10") hoặc đối tượng datetime/Timestamp.
        - Gọi cho Thành viên C: dùng ngày cuối dữ liệu (hoặc +1 ngày).
        - Gọi cho Thành viên D: dùng mốc cắt sớm hơn, đã thống nhất với D.
    cot_customer_id : str, mặc định "Customer ID"
        Tên cột định danh khách hàng.
    cot_invoice : str, mặc định "Invoice"
        Tên cột định danh hoá đơn.
    cot_ngay : str, mặc định "InvoiceDate"
        Tên cột ngày giờ giao dịch.
    cot_total_price : str, mặc định "TotalPrice"
        Tên cột giá trị giao dịch (= Quantity * Price). Cột này cần được
        tính trước bằng hàm tinh_total_price() trong feature_engineering.py.

    Returns
    -------
    pandas.DataFrame
        DataFrame với các cột: "Customer ID", "Recency", "Frequency",
        "Monetary" - đúng tên cột đã thống nhất với nhóm để Thành viên C
        và D dùng nhất quán.

    Ví dụ sử dụng
    -------------
    >>> # Lần 1: tính RFM toàn kỳ cho Thành viên C (Clustering)
    >>> rfm_cho_C = tinh_rfm(df_sach, snapshot_date="2011-12-10")
    >>>
    >>> # Lần 2: tính RFM cắt sớm hơn cho Thành viên D (Churn) - mốc do D đề xuất
    >>> rfm_cho_D = tinh_rfm(df_sach, snapshot_date="2011-09-01")
    """
    df_tinh = df.copy()

    # Chuẩn hoá snapshot_date về đúng kiểu Timestamp để so sánh chính xác
    snapshot_date = pd.Timestamp(snapshot_date)

    # Đảm bảo cột ngày tháng đúng kiểu datetime trước khi so sánh/tính toán
    df_tinh[cot_ngay] = pd.to_datetime(df_tinh[cot_ngay])

    # BƯỚC 1: loại các dòng thiếu Customer ID - không thể tính RFM cho
    # khách hàng không xác định được danh tính.
    so_dong_truoc = len(df_tinh)
    df_tinh = df_tinh[df_tinh[cot_customer_id].notna()].copy()
    print(f"[tinh_rfm] Đã loại {so_dong_truoc - len(df_tinh):,} dòng thiếu "
          f"'{cot_customer_id}' (còn lại {len(df_tinh):,} dòng để tính RFM).")

    # BƯỚC 2 - CỰC KỲ QUAN TRỌNG: chỉ giữ lại giao dịch xảy ra TRƯỚC HOẶC
    # BẰNG snapshot_date. Đây là dòng code trực tiếp ngăn chặn rò rỉ dữ liệu
    # (data leakage) khi hàm này được gọi với snapshot_date cắt sớm cho D.
    so_dong_truoc_loc_ngay = len(df_tinh)
    df_tinh = df_tinh[df_tinh[cot_ngay] <= snapshot_date].copy()
    so_dong_bi_loai_do_sau_snapshot = so_dong_truoc_loc_ngay - len(df_tinh)

    print(f"[tinh_rfm] Mốc snapshot_date = {snapshot_date.date()}. "
          f"Đã loại {so_dong_bi_loai_do_sau_snapshot:,} dòng có ngày giao dịch "
          f"SAU mốc này (còn lại {len(df_tinh):,} dòng).")

    if so_dong_bi_loai_do_sau_snapshot == 0:
        print("  (Lưu ý: 0 dòng bị loại nghĩa là snapshot_date đang lớn hơn "
              "hoặc bằng ngày giao dịch cuối cùng trong dữ liệu - đúng như "
              "kỳ vọng khi tính RFM 'toàn kỳ' cho Thành viên C.)")

    # BƯỚC 3: gộp nhóm theo Customer ID, tính Recency/Frequency/Monetary
    bang_rfm = df_tinh.groupby(cot_customer_id).agg(
        NgayMuaGanNhat=(cot_ngay, "max"),
        Frequency=(cot_invoice, "nunique"),
        Monetary=(cot_total_price, "sum"),
    ).reset_index()

    # Recency = số ngày từ lần mua gần nhất đến snapshot_date
    bang_rfm["Recency"] = (snapshot_date - bang_rfm["NgayMuaGanNhat"]).dt.days

    # Sắp xếp lại cột theo đúng thứ tự chuẩn đã thống nhất với nhóm
    bang_rfm = bang_rfm[[cot_customer_id, "Recency", "Frequency", "Monetary"]]

    print(f"[tinh_rfm] Đã tính RFM cho {len(bang_rfm):,} khách hàng.")

    return bang_rfm


# ==============================================================================
# KHỐI TEST NHANH - chỉ chạy khi gọi trực tiếp "python rfm_calculator.py"
# ==============================================================================
if __name__ == "__main__":
    print("TEST NHANH MODULE rfm_calculator.py (dùng dữ liệu giả lập)\n")

    # Dữ liệu giả lập: 2 khách hàng, có giao dịch trải dài qua nhiều ngày,
    # trong đó có giao dịch xảy ra SAU mốc snapshot dự kiến dùng để test
    # cơ chế chống rò rỉ dữ liệu.
    df_test = pd.DataFrame({
        "Customer ID": [1001, 1001, 1001, 1002, 1002, None],
        "Invoice":     ["A1", "A2", "A3", "B1", "B2", "C1"],
        "InvoiceDate": pd.to_datetime([
            "2011-01-01", "2011-03-01", "2011-09-15",  # khách 1001
            "2011-02-01", "2011-06-01",                 # khách 1002
            "2011-05-01",                                # thiếu Customer ID
        ]),
        "TotalPrice": [100.0, 150.0, 200.0, 50.0, 80.0, 999.0],
    })

    print("Dữ liệu giả lập ban đầu:")
    print(df_test)

    print("\n[Test 1] Tính RFM với snapshot_date = '2011-12-31' (toàn kỳ, giống Thành viên C)")
    rfm_toan_ky = tinh_rfm(df_test, snapshot_date="2011-12-31")
    print(rfm_toan_ky)
    # Khách 1001: Recency = (2011-12-31 - 2011-09-15).days = 107, Frequency=3, Monetary=450
    assert rfm_toan_ky.loc[rfm_toan_ky["Customer ID"] == 1001, "Frequency"].iloc[0] == 3
    assert rfm_toan_ky.loc[rfm_toan_ky["Customer ID"] == 1001, "Monetary"].iloc[0] == 450.0
    print("✔ RFM toàn kỳ tính đúng.")

    print("\n[Test 2] Tính RFM với snapshot_date = '2011-04-01' (cắt sớm, giống Thành viên D)")
    rfm_cat_som = tinh_rfm(df_test, snapshot_date="2011-04-01")
    print(rfm_cat_som)
    # Khách 1001: chỉ còn 2 giao dịch (01/01, 01/03), giao dịch 09/15 bị loại
    # vì xảy ra SAU snapshot -> Frequency phải giảm từ 3 xuống 2, đây chính
    # là bằng chứng cơ chế chống rò rỉ dữ liệu hoạt động đúng.
    assert rfm_cat_som.loc[rfm_cat_som["Customer ID"] == 1001, "Frequency"].iloc[0] == 2
    assert rfm_cat_som.loc[rfm_cat_som["Customer ID"] == 1001, "Monetary"].iloc[0] == 250.0
    print("✔ RFM cắt tại snapshot tính đúng - giao dịch SAU snapshot đã bị loại "
          "(Frequency giảm từ 3 xuống 2 đúng như kỳ vọng).")
    print("✔ Cơ chế CHỐNG RÒ RỈ DỮ LIỆU hoạt động chính xác.")

    print("\n✔ TEST THÀNH CÔNG toàn bộ module rfm_calculator.py")