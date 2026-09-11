"""
==============================================================================
MODULE: cleaning.py
TÁC DỤNG: Chứa các hàm LÀM SẠCH dữ liệu giao dịch - dựa trên các phát hiện
          đã ghi nhận ở notebook 01_data_understanding.ipynb. Đây là file
          quan trọng thứ hai của Thành viên A (sau notebook 01) - vì rubric
          Midterm yêu cầu "minh chứng code thật" cho phần tiền xử lý.

NGƯỜI PHỤ TRÁCH: Thành viên A - Data & Infrastructure Lead

NGUYÊN TẮC THIẾT KẾ:
- MỖI HÀM TRẢ VỀ DATAFRAME MỚI, không sửa in-place trên DataFrame gốc
  truyền vào -> giúp dễ debug (so sánh trước/sau), tránh lỗi ngầm khó phát
  hiện khi nhiều hàm cùng thao tác trên 1 DataFrame.
- Mỗi hàm IN RA số dòng đã loại bỏ -> giúp notebook 02 hiển thị log rõ ràng
  cho từng bước làm sạch.
- Danh sách StockCode phi sản phẩm được đặt thành biến riêng, có comment
  giải thích từng mã, KHÔNG hard-code rải rác trong hàm.
==============================================================================
"""

import numpy as np
import pandas as pd


# ==============================================================================
# 1. CÁC HẰNG SỐ / DANH SÁCH DÙNG CHUNG - đặt riêng, có giải thích rõ ràng
# ==============================================================================

# Danh sách StockCode KHÔNG phải sản phẩm thật (phí, chiết khấu, nghiệp vụ kế toán,
# dữ liệu test nội bộ...) - đã được xác nhận trực tiếp trên dữ liệu thật ở notebook 01.
MA_PHI_SAN_PHAM = {
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
    "TEST001":       "Dữ liệu test nội bộ, không phải sản phẩm thật",
    "TEST002":       "Dữ liệu test nội bộ, không phải sản phẩm thật",
}

# Các từ khoá trong cột Description cho biết đây là bút toán điều chỉnh
# kho (hàng thất lạc/hư hỏng), KHÔNG phải giao dịch bán hàng thật.
# Đã xác nhận thực tế: các dòng này luôn đi kèm Price = 0 và thiếu Customer ID.
TU_KHOA_DIEU_CHINH_KHO = [
    "LOST", "DAMAGE", "DAMAGES", "FOUND", "MISSING", "THROWN AWAY",
    "WRONG", "SMASHED", "CRUSHED", "MOULDY", "RUSTY", "CHECK",
]


# ==============================================================================
# 2. HÀM LOẠI BỎ INVOICE HUỶ (Invoice bắt đầu bằng "C")
# ==============================================================================

def loai_invoice_huy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loại bỏ các dòng thuộc INVOICE HUỶ - tức Invoice có mã bắt đầu bằng
    chữ "C" (cancellation), theo đúng quy ước của dataset Online Retail II.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame giao dịch gốc, cần có cột "Invoice".

    Returns
    -------
    pandas.DataFrame
        DataFrame MỚI đã loại bỏ các dòng invoice huỷ (không sửa df gốc).
    """
    df_moi = df.copy()

    invoice_dang_chuoi = df_moi["Invoice"].astype(str)
    mat_na_giu_lai = ~invoice_dang_chuoi.str.startswith("C")

    so_dong_loai_bo = (~mat_na_giu_lai).sum()
    print(f"[loai_invoice_huy] Đã loại {so_dong_loai_bo:,} dòng invoice huỷ "
          f"(còn lại {mat_na_giu_lai.sum():,} dòng).")

    return df_moi[mat_na_giu_lai].reset_index(drop=True)


def loai_invoice_dieu_chinh_no_xau(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loại bỏ các dòng thuộc bút toán "Adjust bad debt" (xoá nợ xấu kế toán)
    - đây là các Invoice có mã bắt đầu bằng chữ "A" (khác với "C" của invoice
    huỷ), phát hiện thực tế trên dữ liệu: StockCode = "B", Price cực âm,
    KHÔNG phải giao dịch bán hàng.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame giao dịch, cần có cột "Invoice".

    Returns
    -------
    pandas.DataFrame
        DataFrame MỚI đã loại bỏ các dòng điều chỉnh nợ xấu.
    """
    df_moi = df.copy()

    invoice_dang_chuoi = df_moi["Invoice"].astype(str)
    mat_na_giu_lai = ~invoice_dang_chuoi.str.startswith("A")

    so_dong_loai_bo = (~mat_na_giu_lai).sum()
    print(f"[loai_invoice_dieu_chinh_no_xau] Đã loại {so_dong_loai_bo:,} dòng "
          f"'Adjust bad debt' (còn lại {mat_na_giu_lai.sum():,} dòng).")

    return df_moi[mat_na_giu_lai].reset_index(drop=True)


# ==============================================================================
# 3. HÀM LOẠI BỎ QUANTITY ÂM BẤT THƯỜNG (không phải invoice huỷ)
# ==============================================================================

def loai_dieu_chinh_kho_bat_thuong(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loại bỏ các dòng "Quantity âm nhưng KHÔNG thuộc invoice huỷ" - đây là
    các bút toán điều chỉnh sổ sách nội bộ (hàng thất lạc, hư hỏng...),
    đã xác nhận thực tế luôn có Price = 0 và thiếu Customer ID.

    LƯU Ý: hàm này nên gọi SAU khi đã gọi loai_invoice_huy() và
    loai_invoice_dieu_chinh_no_xau(), vì logic bên dưới giả định các dòng
    Invoice "C" và "A" đã được loại bỏ trước đó - phần Quantity âm còn sót
    lại ở bước này chính là nhóm bút toán điều chỉnh kho cần loại tiếp.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame giao dịch, cần có cột "Quantity".

    Returns
    -------
    pandas.DataFrame
        DataFrame MỚI đã loại bỏ các dòng điều chỉnh kho bất thường.
    """
    df_moi = df.copy()

    mat_na_giu_lai = df_moi["Quantity"] >= 0

    so_dong_loai_bo = (~mat_na_giu_lai).sum()
    print(f"[loai_dieu_chinh_kho_bat_thuong] Đã loại {so_dong_loai_bo:,} dòng "
          f"Quantity âm bất thường (còn lại {mat_na_giu_lai.sum():,} dòng).")

    return df_moi[mat_na_giu_lai].reset_index(drop=True)


# ==============================================================================
# 4. HÀM LOẠI BỎ STOCKCODE PHI SẢN PHẨM
# ==============================================================================

def loai_stockcode_phi_san_pham(df: pd.DataFrame,
                                  danh_sach_ma: dict = None) -> pd.DataFrame:
    """
    Loại bỏ các dòng có StockCode KHÔNG phải sản phẩm thật (phí, chiết
    khấu, nghiệp vụ kế toán, dữ liệu test...) - theo danh sách MA_PHI_SAN_PHAM
    đã xác nhận thực tế ở notebook 01.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame giao dịch, cần có cột "StockCode".
    danh_sach_ma : dict, tuỳ chọn
        Dictionary {mã: giải thích} các StockCode cần loại bỏ. Nếu để None
        (mặc định), dùng danh sách chuẩn MA_PHI_SAN_PHAM đã định nghĩa ở
        đầu file này.

    Returns
    -------
    pandas.DataFrame
        DataFrame MỚI đã loại bỏ các dòng StockCode phi sản phẩm.
    """
    if danh_sach_ma is None:
        danh_sach_ma = MA_PHI_SAN_PHAM

    df_moi = df.copy()

    ma_dang_hoa = df_moi["StockCode"].astype(str).str.upper()
    mat_na_giu_lai = ~ma_dang_hoa.isin(danh_sach_ma.keys())

    so_dong_loai_bo = (~mat_na_giu_lai).sum()
    print(f"[loai_stockcode_phi_san_pham] Đã loại {so_dong_loai_bo:,} dòng "
          f"StockCode phi sản phẩm (còn lại {mat_na_giu_lai.sum():,} dòng).")

    return df_moi[mat_na_giu_lai].reset_index(drop=True)


# ==============================================================================
# 5. HÀM LOẠI BỎ DÒNG TRÙNG LẶP HOÀN TOÀN
# ==============================================================================

def loai_duplicate(df: pd.DataFrame, cot_bo_qua: list = None) -> pd.DataFrame:
    """
    Loại bỏ các dòng trùng lặp hoàn toàn, chỉ giữ lại 1 bản ghi duy nhất
    cho mỗi nhóm trùng lặp.

    LƯU Ý KỸ THUẬT QUAN TRỌNG: nếu DataFrame có thêm cột phụ để đánh dấu
    nguồn gốc (ví dụ cột "Sheet" do hàm load_raw_excel() tự thêm vào), cột
    này CẦN được loại ra khỏi việc so sánh trùng lặp - nếu không, 2 dòng dữ
    liệu giống hệt nhau nhưng khác nguồn (khác sheet) sẽ KHÔNG bị coi là
    trùng lặp, dẫn đến bỏ sót duplicate thật sự tồn tại trong dữ liệu gốc.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame giao dịch cần loại duplicate.
    cot_bo_qua : list[str], tuỳ chọn
        Danh sách tên cột KHÔNG tham gia vào việc so sánh trùng lặp (ví dụ
        cột "Sheet"). Nếu để None (mặc định), tự động bỏ qua cột "Sheet"
        nếu cột này tồn tại trong DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame MỚI đã loại bỏ dòng trùng lặp.
    """
    df_moi = df.copy()

    if cot_bo_qua is None:
        cot_bo_qua = ["Sheet"] if "Sheet" in df_moi.columns else []

    cac_cot_so_sanh = [c for c in df_moi.columns if c not in cot_bo_qua]

    so_dong_truoc = len(df_moi)
    df_moi = df_moi.drop_duplicates(subset=cac_cot_so_sanh).reset_index(drop=True)
    so_dong_loai_bo = so_dong_truoc - len(df_moi)

    print(f"[loai_duplicate] Đã loại {so_dong_loai_bo:,} dòng trùng lặp "
          f"(còn lại {len(df_moi):,} dòng).")

    return df_moi


# ==============================================================================
# 6. HÀM XỬ LÝ OUTLIER (Quantity/Price)
# ==============================================================================

def xu_ly_outlier(df: pd.DataFrame,
                   cot: str = "Quantity",
                   phuong_phap: str = "iqr",
                   he_so_iqr: float = 1.5,
                   nguong_zscore: float = 3.0,
                   dung_log_truoc_zscore: bool = True) -> pd.DataFrame:
    """
    Loại bỏ outlier trên 1 cột số (thường dùng cho Quantity hoặc Price),
    hỗ trợ 2 phương pháp: IQR hoặc Z-score.

    CĂN CỨ TỪ NOTEBOOK 01 (rất quan trọng, đọc kỹ trước khi dùng):
    - Cột Quantity có skewness RẤT CAO (khoảng 460 ở dữ liệu gốc, so với
      khoảng 1.06 sau log1p-transform) do một số giao dịch bán buôn với
      số lượng cực lớn.
    - Đã xác nhận: nhóm Quantity cao có Price trung vị THẤP HƠN Price
      trung vị chung -> đây là bán buôn hợp lệ (giá ưu đãi số lượng lớn),
      KHÔNG phải lỗi nhập liệu.
    - Vì vậy, mặc định hàm này dùng phương pháp IQR (ổn định hơn với dữ
      liệu lệch) và KHUYẾN NGHỊ chỉ dùng để NHẬN DIỆN/GHI CHÚ outlier,
      cân nhắc kỹ trước khi áp dụng loại bỏ tự động triệt để trên Quantity,
      để tránh xoá nhầm các đơn hàng bán buôn hợp lệ.
    - Nếu dùng Z-score, BẮT BUỘC bật dung_log_truoc_zscore=True (mặc định),
      vì Z-score trên dữ liệu gốc chưa transform sẽ cho kết quả sai lệch
      nghiêm trọng (bắt được quá ít outlier so với thực tế).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame cần xử lý outlier.
    cot : str, mặc định "Quantity"
        Tên cột số cần xử lý outlier.
    phuong_phap : {"iqr", "zscore"}, mặc định "iqr"
        Phương pháp phát hiện outlier.
    he_so_iqr : float, mặc định 1.5
        Hệ số nhân với IQR để xác định cận trên/dưới (dùng khi phuong_phap="iqr").
    nguong_zscore : float, mặc định 3.0
        Ngưỡng |Z-score| để coi là outlier (dùng khi phuong_phap="zscore").
    dung_log_truoc_zscore : bool, mặc định True
        Nếu True và phuong_phap="zscore", áp dụng log1p-transform trước khi
        tính Z-score - theo đúng phát hiện ở notebook 01. CHỈ nên đặt False
        nếu đã tự xác nhận cột dữ liệu không bị lệch (skewness thấp).

    Returns
    -------
    pandas.DataFrame
        DataFrame MỚI đã loại bỏ các dòng outlier theo phương pháp đã chọn.

    Ví dụ sử dụng
    -------------
    >>> # Cách dùng mặc định, khuyến nghị cho Quantity (dữ liệu lệch mạnh)
    >>> df_sach = xu_ly_outlier(df, cot="Quantity", phuong_phap="iqr")
    """
    df_moi = df.copy()

    if phuong_phap == "iqr":
        Q1, Q3 = df_moi[cot].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        can_duoi = Q1 - he_so_iqr * IQR
        can_tren = Q3 + he_so_iqr * IQR
        mat_na_giu_lai = (df_moi[cot] >= can_duoi) & (df_moi[cot] <= can_tren)

    elif phuong_phap == "zscore":
        if dung_log_truoc_zscore:
            # log1p xử lý được cả giá trị 0 (log1p(0) = 0), an toàn hơn log thường
            gia_tri_tinh = np.log1p(df_moi[cot].clip(lower=0))
        else:
            gia_tri_tinh = df_moi[cot]

        trung_binh, do_lech_chuan = gia_tri_tinh.mean(), gia_tri_tinh.std()
        z_score = (gia_tri_tinh - trung_binh) / do_lech_chuan
        mat_na_giu_lai = z_score.abs() <= nguong_zscore

    else:
        raise ValueError(
            f"phuong_phap phải là 'iqr' hoặc 'zscore', nhận được: '{phuong_phap}'"
        )

    so_dong_loai_bo = (~mat_na_giu_lai).sum()
    print(f"[xu_ly_outlier] Cột '{cot}', phương pháp '{phuong_phap}': "
          f"đã loại {so_dong_loai_bo:,} dòng outlier "
          f"(còn lại {mat_na_giu_lai.sum():,} dòng).")

    return df_moi[mat_na_giu_lai].reset_index(drop=True)


# ==============================================================================
# 7. HÀM PIPELINE TỔNG HỢP - gọi lại tuần tự các hàm trên
# ==============================================================================

def lam_sach_du_lieu(df: pd.DataFrame,
                      xu_ly_outlier_quantity: bool = False,
                      phuong_phap_outlier: str = "iqr") -> pd.DataFrame:
    """
    Hàm PIPELINE tổng hợp - gọi lại tuần tự các bước làm sạch theo đúng
    thứ tự đã tổng kết ở Mục 11 của notebook 01_data_understanding.ipynb.

    Thứ tự xử lý (KHÔNG tuỳ ý đảo lộn):
    1. Loại invoice huỷ (Invoice bắt đầu "C")
    2. Loại invoice điều chỉnh nợ xấu (Invoice bắt đầu "A")
    3. Loại Quantity âm bất thường (bút toán điều chỉnh kho)
    4. Loại StockCode phi sản phẩm
    5. Loại dòng trùng lặp hoàn toàn
    6. (Tuỳ chọn) Xử lý outlier Quantity - MẶC ĐỊNH TẮT, vì đã xác nhận
       phần lớn "outlier" Quantity cao là bán buôn hợp lệ, không nên xoá
       tự động trừ khi có lý do rõ ràng.

    QUAN TRỌNG: hàm này KHÔNG loại bỏ các dòng thiếu Customer ID - việc
    này được xử lý RIÊNG ở notebook 02_data_preparation.ipynb, vì mỗi
    pipeline (MBA / RFM / Churn) có yêu cầu khác nhau về Customer ID
    (xem lại Mục 11 của notebook 01: MBA cần GIỮ dòng thiếu Customer ID,
    còn RFM/Churn cần LOẠI các dòng này).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame giao dịch gốc (đã đọc từ load_raw_excel()).
    xu_ly_outlier_quantity : bool, mặc định False
        Nếu True, áp dụng thêm bước xử lý outlier trên cột Quantity.
        Mặc định TẮT vì lý do đã giải thích ở trên.
    phuong_phap_outlier : str, mặc định "iqr"
        Phương pháp xử lý outlier nếu xu_ly_outlier_quantity=True.

    Returns
    -------
    pandas.DataFrame
        DataFrame đã sạch hoàn chỉnh, sẵn sàng cho bước feature engineering
        và tách nhánh theo từng pipeline ở notebook 02.

    Ví dụ sử dụng
    -------------
    >>> from src.utils.io import load_raw_excel
    >>> from src.preprocessing.cleaning import lam_sach_du_lieu
    >>> df_tho = load_raw_excel()
    >>> df_sach = lam_sach_du_lieu(df_tho)
    """
    print("=" * 70)
    print("BẮT ĐẦU PIPELINE LÀM SẠCH DỮ LIỆU")
    print("=" * 70)
    print(f"Số dòng ban đầu: {len(df):,}")
    print("-" * 70)

    df_ket_qua = loai_invoice_huy(df)
    df_ket_qua = loai_invoice_dieu_chinh_no_xau(df_ket_qua)
    df_ket_qua = loai_dieu_chinh_kho_bat_thuong(df_ket_qua)
    df_ket_qua = loai_stockcode_phi_san_pham(df_ket_qua)
    df_ket_qua = loai_duplicate(df_ket_qua)

    if xu_ly_outlier_quantity:
        df_ket_qua = xu_ly_outlier(df_ket_qua, cot="Quantity",
                                     phuong_phap=phuong_phap_outlier)

    print("-" * 70)
    print(f"Số dòng sau khi làm sạch: {len(df_ket_qua):,} "
          f"(đã loại tổng cộng {len(df) - len(df_ket_qua):,} dòng, "
          f"tương đương {(len(df) - len(df_ket_qua)) / len(df) * 100:.2f}%)")
    print("=" * 70)

    return df_ket_qua


# ==============================================================================
# KHỐI TEST NHANH - chỉ chạy khi gọi trực tiếp "python cleaning.py"
# ==============================================================================
if __name__ == "__main__":
    print("TEST NHANH MODULE cleaning.py (dùng dữ liệu giả lập)\n")

    # Tạo DataFrame giả lập nhỏ, cố tình cài các "lỗi" cần được làm sạch
    df_test = pd.DataFrame({
        "Invoice":     ["100001", "100001", "C100002", "A100003", "100004", "100004", "100005"],
        "StockCode":   ["85123A", "85123A", "22423", "B", "POST", "POST", "TEST001"],
        "Description": ["MUG", "MUG", "CANDLE", "Adjust bad debt", "POSTAGE", "POSTAGE", "test"],
        "Quantity":    [6, 6, -2, -1, 1, 1, 5],
        "Price":       [2.5, 2.5, 4.0, -500.0, 18.0, 18.0, 0.0],
        "Customer ID": [12345, 12345, 12345, np.nan, 12346, 12346, 12347],
        "Country":     ["United Kingdom"] * 7,
    })

    print("Dữ liệu giả lập ban đầu:")
    print(df_test)
    print()

    df_da_sach = lam_sach_du_lieu(df_test)

    print("\nDữ liệu sau khi làm sạch:")
    print(df_da_sach)

    # Kỳ vọng: chỉ còn lại 1 dòng "MUG" (đã loại 1 bản duplicate),
    # dòng "CANDLE" (invoice huỷ) bị loại, dòng "Adjust bad debt" bị loại,
    # dòng "POSTAGE" (StockCode phi sản phẩm) bị loại,
    # dòng "TEST001" (StockCode phi sản phẩm) bị loại.
    assert len(df_da_sach) == 1, "Test thất bại - kiểm tra lại logic các hàm!"
    print("\n✔ TEST THÀNH CÔNG: pipeline làm sạch hoạt động đúng như kỳ vọng.")