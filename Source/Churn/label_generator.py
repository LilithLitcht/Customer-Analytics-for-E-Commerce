"""
==============================================================================
MODULE: label_generator.py
TÁC DỤNG: Sinh NHÃN CHURN (0/1) cho bài toán phân lớp dự đoán khách hàng rời
          bỏ, dựa trên phân tích thống kê khoảng cách giữa các lần mua liên
          tiếp (inter-purchase gap) thay vì chọn ngưỡng tuỳ tiện.

NGƯỜI PHỤ TRÁCH: Thành viên D - Classification Lead (Churn)

==============================================================================
VÌ SAO FILE NÀY QUAN TRỌNG NHẤT TOÀN DỰ ÁN VỀ MẶT PHƯƠNG PHÁP LUẬN
==============================================================================
Khác với Market Basket Analysis (Thành viên B) hay Clustering (Thành viên C),
bài toán Churn KHÔNG có sẵn nhãn trong dữ liệu gốc. Dataset Online Retail II
chỉ ghi nhận giao dịch, không có cột nào nói "khách hàng này đã rời bỏ".

Vì vậy, nhóm phải TỰ ĐỊNH NGHĨA thế nào là "churn" - và đây chính là quyết
định dễ bị phản biện nhất khi bảo vệ đồ án. Câu hỏi giảng viên chắc chắn hỏi:
"Tại sao chọn cửa sổ churn là X ngày mà không phải 30 hay 90 ngày?"

Câu trả lời của nhóm KHÔNG được là "vì thấy người ta hay dùng 90 ngày", mà
phải dựa trên PHÂN PHỐI THỰC TẾ của dữ liệu: nếu 75% các lần mua liên tiếp
của khách hàng cách nhau không quá W ngày, thì một khách hàng im lặng quá W
ngày là bất thường so với chính hành vi của tập khách hàng đó - đó mới là
căn cứ thống kê.

==============================================================================
ĐỊNH NGHĨA CHURN DÙNG TRONG DỰ ÁN
==============================================================================
Với một mốc thời gian tham chiếu (snapshot_date) và một độ dài cửa sổ quan
sát (window_days):

  - Tập khách hàng xét đến: mọi khách có ÍT NHẤT 1 giao dịch TRƯỚC HOẶC BẰNG
    snapshot_date (tức là đã từng hoạt động tại thời điểm ta đứng dự đoán).
  - Churn = 1: khách KHÔNG phát sinh giao dịch nào trong khoảng
    (snapshot_date, snapshot_date + window_days].
  - Churn = 0: khách CÓ quay lại mua trong khoảng đó.

Điểm mấu chốt chống rò rỉ dữ liệu: đặc trưng đầu vào (RFM) được tính TẠI
snapshot_date (chỉ dùng dữ liệu quá khứ), còn nhãn được xác định từ dữ liệu
SAU snapshot_date. Hai phần này KHÔNG bao giờ được trộn lẫn.

QUY ƯỚC ĐẶT TÊN: tên hàm/biến/tham số dùng TIẾNG ANH chuẩn PEP8, comment và
docstring dùng TIẾNG VIỆT - đồng bộ với các file khác trong dự án.
==============================================================================
"""

import numpy as np
import pandas as pd


# ==============================================================================
# 1. TÍNH KHOẢNG CÁCH GIỮA CÁC LẦN MUA LIÊN TIẾP (INTER-PURCHASE GAP)
# ==============================================================================

def compute_inter_purchase_gaps(df: pd.DataFrame,
                                  customer_id_col: str = "Customer ID",
                                  date_col: str = "InvoiceDate") -> pd.Series:
    """
    Tính khoảng cách (tính bằng NGÀY) giữa hai lần mua liên tiếp của cùng một
    khách hàng, gộp lại thành một chuỗi duy nhất cho toàn bộ tập khách hàng.

    LƯU Ý KỸ THUẬT: hàm chuẩn hoá thời gian về mốc NGÀY (bỏ phần giờ/phút)
    trước khi tính, vì hai hoá đơn phát sinh cách nhau vài phút trong cùng một
    ngày thực chất là CÙNG MỘT LẦN MUA SẮM - nếu không chuẩn hoá, chúng sẽ tạo
    ra vô số "gap = 0 ngày" giả tạo, kéo lệch toàn bộ phân phối xuống thấp.

    Parameters
    ----------
    df : pandas.DataFrame
        Dữ liệu giao dịch đã làm sạch, cần có cột Customer ID và InvoiceDate.
        Các dòng thiếu Customer ID sẽ tự động bị loại.
    customer_id_col : str, mặc định "Customer ID"
        Tên cột định danh khách hàng.
    date_col : str, mặc định "InvoiceDate"
        Tên cột ngày giờ giao dịch.

    Returns
    -------
    pandas.Series
        Chuỗi các khoảng cách (số ngày) giữa hai lần mua liên tiếp, gộp từ
        MỌI khách hàng có ít nhất 2 ngày mua khác nhau. Khách chỉ mua đúng
        1 lần sẽ không đóng góp giá trị nào (vì không có "khoảng cách").

    Ví dụ sử dụng
    -------------
    >>> gaps = compute_inter_purchase_gaps(transactions_df)
    >>> gaps.quantile(0.75)
    70.0
    """
    working_df = df[df[customer_id_col].notna()].copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col])

    # Chuẩn hoá về mốc ngày (00:00) để gộp các hoá đơn trong cùng một ngày
    working_df["PurchaseDate"] = working_df[date_col].dt.normalize()

    # Với mỗi khách: lấy danh sách ngày mua DUY NHẤT, sắp xếp tăng dần
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

    print(f"[compute_inter_purchase_gaps] Thu được {len(gaps_series):,} khoảng cách "
          f"từ {n_repeat_customers:,} khách mua lặp lại.")
    print(f"[compute_inter_purchase_gaps] Có {n_one_time_customers:,} khách chỉ mua ĐÚNG 1 LẦN "
          f"({n_one_time_customers / len(purchase_dates_per_customer) * 100:.1f}%) - "
          f"nhóm này không có khoảng cách để tính, cần xử lý riêng khi sinh nhãn.")

    return gaps_series


def summarize_gap_distribution(gaps: pd.Series,
                                percentiles: list = None) -> pd.DataFrame:
    """
    Lập bảng tóm tắt phân phối inter-purchase gap theo các mốc percentile -
    đây là BẢNG CĂN CỨ để chọn độ dài cửa sổ churn, và cũng là bằng chứng
    trực tiếp dùng khi trả lời phản biện.

    Parameters
    ----------
    gaps : pandas.Series
        Chuỗi khoảng cách trả về từ compute_inter_purchase_gaps().
    percentiles : list[int], tuỳ chọn
        Danh sách các mốc percentile cần tính. Mặc định:
        [50, 60, 70, 75, 80, 85, 90, 95].

    Returns
    -------
    pandas.DataFrame
        Bảng gồm 3 cột: "percentile", "gap_days", và "y_nghia" (diễn giải
        bằng lời cho từng mốc).

    Ví dụ sử dụng
    -------------
    >>> gap_summary = summarize_gap_distribution(gaps)
    """
    if percentiles is None:
        percentiles = [50, 60, 70, 75, 80, 85, 90, 95]

    records = []
    for p in percentiles:
        gap_value = gaps.quantile(p / 100)
        records.append({
            "percentile": f"P{p}",
            "gap_days": round(gap_value, 1),
            "y_nghia": f"{p}% số lần mua liên tiếp cách nhau không quá {gap_value:.0f} ngày",
        })

    return pd.DataFrame(records)


def suggest_churn_window(gaps: pd.Series, percentile: int = 75) -> int:
    """
    Đề xuất độ dài cửa sổ churn (số ngày) dựa trên một mốc percentile của
    phân phối inter-purchase gap.

    LÝ GIẢI CHỌN PERCENTILE 75 LÀM MẶC ĐỊNH (nội dung cần thuộc để phản biện):
      - Nếu chọn percentile QUÁ THẤP (ví dụ P50): cửa sổ ngắn, rất nhiều khách
        hàng bình thường (vốn có chu kỳ mua dài hơn trung vị) sẽ bị gán nhầm
        là churn -> nhãn nhiễu, mô hình học sai.
      - Nếu chọn percentile QUÁ CAO (ví dụ P95): cửa sổ quá dài, gần như không
        ai bị coi là churn -> nhãn mất cân bằng cực đoan, và quan trọng hơn là
        doanh nghiệp phát hiện quá muộn, không còn kịp giữ chân khách.
      - P75 là điểm cân bằng: một khách im lặng lâu hơn 75% các chu kỳ mua
        thông thường đã đủ bất thường để đáng cảnh báo, nhưng chưa quá khắt
        khe tới mức gắn nhãn sai hàng loạt.

    Đây là giá trị ĐỀ XUẤT, không phải bắt buộc - Thành viên D nên thử vài mốc
    percentile khác nhau và xem xét cả yếu tố nghiệp vụ trước khi chốt.

    Parameters
    ----------
    gaps : pandas.Series
        Chuỗi khoảng cách trả về từ compute_inter_purchase_gaps().
    percentile : int, mặc định 75
        Mốc percentile dùng để lấy ngưỡng.

    Returns
    -------
    int
        Độ dài cửa sổ churn đề xuất, tính bằng số ngày (làm tròn xuống số nguyên).

    Ví dụ sử dụng
    -------------
    >>> window_days = suggest_churn_window(gaps, percentile=75)
    >>> window_days
    70
    """
    window_days = int(gaps.quantile(percentile / 100))

    print(f"[suggest_churn_window] Cửa sổ churn đề xuất theo P{percentile}: "
          f"{window_days} ngày.")
    print(f"  Diễn giải: khách hàng không quay lại mua trong {window_days} ngày "
          f"được coi là đã rời bỏ, vì {percentile}% các chu kỳ mua thực tế "
          f"đều ngắn hơn khoảng này.")

    return window_days


# ==============================================================================
# 2. SINH NHÃN CHURN TẠI MỘT MỐC SNAPSHOT
# ==============================================================================

def generate_churn_labels(df: pd.DataFrame,
                           snapshot_date,
                           window_days: int,
                           customer_id_col: str = "Customer ID",
                           date_col: str = "InvoiceDate") -> pd.DataFrame:
    """
    Sinh nhãn churn (0/1) cho từng khách hàng tại một mốc snapshot cho trước.

    Quy tắc gán nhãn:
      - Chỉ xét những khách đã có giao dịch TRƯỚC HOẶC BẰNG snapshot_date
        (khách chưa từng mua tại thời điểm đó thì không có gì để dự đoán).
      - Churn = 1 nếu khách KHÔNG có giao dịch nào trong khoảng
        (snapshot_date, snapshot_date + window_days].
      - Churn = 0 nếu khách CÓ quay lại mua trong khoảng đó.

    CẢNH BÁO VỀ TÍNH HỢP LỆ CỦA CỬA SỔ QUAN SÁT: hàm sẽ cảnh báo nếu
    snapshot_date + window_days vượt quá ngày cuối cùng có trong dữ liệu -
    khi đó cửa sổ quan sát bị "cụt", nhãn churn sẽ bị thiên lệch (nhiều khách
    bị gán churn oan chỉ vì dữ liệu kết thúc trước khi họ kịp quay lại).

    Parameters
    ----------
    df : pandas.DataFrame
        Dữ liệu giao dịch đã làm sạch.
    snapshot_date : str hoặc datetime hoặc pandas.Timestamp
        Mốc thời gian đứng dự đoán.
    window_days : int
        Độ dài cửa sổ quan sát sau snapshot, tính bằng ngày.
    customer_id_col : str, mặc định "Customer ID"
        Tên cột định danh khách hàng.
    date_col : str, mặc định "InvoiceDate"
        Tên cột ngày giờ giao dịch.

    Returns
    -------
    pandas.DataFrame
        Bảng gồm 2 cột: "Customer ID" và "Churn" (0 hoặc 1).

    Ví dụ sử dụng
    -------------
    >>> labels = generate_churn_labels(transactions_df,
    ...                                 snapshot_date="2011-09-30",
    ...                                 window_days=70)
    >>> labels["Churn"].mean()
    0.606
    """
    working_df = df[df[customer_id_col].notna()].copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col])

    snapshot_timestamp = pd.Timestamp(snapshot_date)
    window_end = snapshot_timestamp + pd.Timedelta(days=window_days)

    # Kiểm tra cửa sổ quan sát có bị cụt không
    last_date_in_data = working_df[date_col].max()
    if window_end > last_date_in_data:
        n_days_short = (window_end - last_date_in_data).days
        print(f"[generate_churn_labels] CẢNH BÁO: cửa sổ quan sát kết thúc ngày "
              f"{window_end.date()} nhưng dữ liệu chỉ có đến {last_date_in_data.date()} "
              f"(thiếu {n_days_short} ngày). Nhãn churn có thể bị thiên lệch "
              f"(một số khách bị gán churn chỉ vì dữ liệu kết thúc sớm).")

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
    # Churn = 1 khi khách KHÔNG nằm trong danh sách quay lại mua
    labels_df["Churn"] = (~labels_df[customer_id_col].isin(customers_returned)).astype(int)

    churn_rate = labels_df["Churn"].mean()
    print(f"[generate_churn_labels] snapshot={snapshot_timestamp.date()}, "
          f"cửa sổ={window_days} ngày -> {len(labels_df):,} khách hàng, "
          f"churn={labels_df['Churn'].sum():,} ({churn_rate * 100:.1f}%), "
          f"không churn={(labels_df['Churn'] == 0).sum():,}")

    return labels_df


# ==============================================================================
# 3. TẠO 2 MỐC SNAPSHOT ĐỂ TÁCH TRAIN/TEST THEO THỜI GIAN
# ==============================================================================

def build_time_based_snapshots(df: pd.DataFrame,
                                 window_days: int,
                                 date_col: str = "InvoiceDate") -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Tính ra HAI mốc snapshot liên tiếp, dùng để tách tập huấn luyện và tập
    kiểm thử THEO THỜI GIAN (thay vì chia ngẫu nhiên).

    VÌ SAO KHÔNG DÙNG train_test_split NGẪU NHIÊN (câu hỏi phản biện kinh điển):
    Bài toán churn về bản chất là dự đoán TƯƠNG LAI từ QUÁ KHỨ. Nếu chia ngẫu
    nhiên, tập test sẽ chứa những khách hàng thuộc cùng một khoảng thời gian
    với tập train - mô hình vô tình được "học ké" bối cảnh thị trường của
    chính giai đoạn mà nó phải dự đoán (ví dụ: đợt khuyến mãi lớn, mùa cao
    điểm Giáng Sinh). Kết quả đánh giá khi đó lạc quan giả tạo, và mô hình sẽ
    hoạt động kém khi triển khai thật.

    Cách tách theo thời gian ở đây mô phỏng đúng tình huống thực tế: đứng ở
    thời điểm T_train để học, rồi kiểm tra xem mô hình dự đoán tốt không ở
    thời điểm T_test muộn hơn - hai cửa sổ quan sát KHÔNG chồng lấn nhau.

    Bố trí thời gian (giả sử ngày cuối dữ liệu là D):
        T_test  = D - window_days       -> cửa sổ quan sát: (T_test, D]
        T_train = T_test - window_days  -> cửa sổ quan sát: (T_train, T_test]

    Parameters
    ----------
    df : pandas.DataFrame
        Dữ liệu giao dịch đã làm sạch.
    window_days : int
        Độ dài cửa sổ churn đã chốt.
    date_col : str, mặc định "InvoiceDate"
        Tên cột ngày giờ giao dịch.

    Returns
    -------
    tuple[pandas.Timestamp, pandas.Timestamp]
        (train_snapshot, test_snapshot) - hai mốc snapshot theo thứ tự thời gian.

    Ví dụ sử dụng
    -------------
    >>> train_snapshot, test_snapshot = build_time_based_snapshots(tx_df, window_days=70)
    """
    date_series = pd.to_datetime(df[date_col])
    last_date = date_series.max().normalize()

    test_snapshot = last_date - pd.Timedelta(days=window_days)
    train_snapshot = test_snapshot - pd.Timedelta(days=window_days)

    print(f"[build_time_based_snapshots] Ngày cuối dữ liệu: {last_date.date()}")
    print(f"  TRAIN: snapshot={train_snapshot.date()}, "
          f"cửa sổ quan sát nhãn = ({train_snapshot.date()}, {test_snapshot.date()}]")
    print(f"  TEST : snapshot={test_snapshot.date()}, "
          f"cửa sổ quan sát nhãn = ({test_snapshot.date()}, {last_date.date()}]")
    print("  -> Hai cửa sổ KHÔNG chồng lấn nhau, đảm bảo mô hình học quá khứ "
          "và được kiểm thử trên tương lai.")

    # Kiểm tra mốc train có đủ dữ liệu lịch sử phía trước không
    first_date = date_series.min().normalize()
    history_days = (train_snapshot - first_date).days
    if history_days < window_days:
        print(f"  CẢNH BÁO: chỉ có {history_days} ngày lịch sử trước mốc TRAIN - "
              f"có thể không đủ để tính đặc trưng RFM đáng tin cậy.")

    return train_snapshot, test_snapshot


# ==============================================================================
# 4. BỔ SUNG ĐẶC TRƯNG PHÁI SINH CHO BÀI TOÁN CHURN
# ==============================================================================

def add_churn_features(rfm_df: pd.DataFrame,
                        transactions_df: pd.DataFrame,
                        snapshot_date,
                        customer_id_col: str = "Customer ID",
                        date_col: str = "InvoiceDate") -> pd.DataFrame:
    """
    Bổ sung một số đặc trưng phái sinh ngoài R/F/M cơ bản, giúp mô hình phân
    lớp có thêm thông tin để phân biệt khách sắp rời bỏ.

    Các đặc trưng được thêm:
      - AvgOrderValue : giá trị trung bình mỗi hoá đơn (= Monetary / Frequency).
                        Phân biệt khách "mua nhiều lần giá trị nhỏ" với khách
                        "mua ít lần nhưng giá trị lớn".
      - Tenure        : số ngày kể từ lần mua ĐẦU TIÊN đến snapshot. Phân biệt
                        khách mới (tenure ngắn) với khách lâu năm.
      - AvgGapDays    : khoảng cách trung bình giữa các lần mua của riêng khách
                        đó. Đây là đặc trưng mạnh vì nó cho biết nhịp mua
                        RIÊNG của từng khách, thay vì so với mặt bằng chung.

    QUAN TRỌNG - CHỐNG RÒ RỈ DỮ LIỆU: mọi đặc trưng ở đây đều chỉ dùng giao
    dịch xảy ra TRƯỚC HOẶC BẰNG snapshot_date, giống hệt nguyên tắc đã áp dụng
    khi tính RFM.

    Parameters
    ----------
    rfm_df : pandas.DataFrame
        Bảng RFM đã tính tại đúng snapshot_date này.
    transactions_df : pandas.DataFrame
        Dữ liệu giao dịch đã làm sạch (để tính Tenure và AvgGapDays).
    snapshot_date : str hoặc datetime hoặc pandas.Timestamp
        Mốc snapshot - phải TRÙNG với mốc đã dùng khi tính rfm_df.
    customer_id_col : str, mặc định "Customer ID"
        Tên cột định danh khách hàng.
    date_col : str, mặc định "InvoiceDate"
        Tên cột ngày giờ giao dịch.

    Returns
    -------
    pandas.DataFrame
        Bảng RFM ban đầu, bổ sung thêm 3 cột: AvgOrderValue, Tenure, AvgGapDays.

    Ví dụ sử dụng
    -------------
    >>> features = add_churn_features(rfm_df, tx_df, snapshot_date="2011-09-30")
    """
    snapshot_timestamp = pd.Timestamp(snapshot_date)

    working_df = transactions_df[transactions_df[customer_id_col].notna()].copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col])
    # Chỉ dùng dữ liệu quá khứ - đây là rào chắn chống rò rỉ dữ liệu
    working_df = working_df[working_df[date_col] <= snapshot_timestamp]
    working_df["PurchaseDate"] = working_df[date_col].dt.normalize()

    result_df = rfm_df.copy()

    # Đặc trưng 1: giá trị trung bình mỗi hoá đơn
    # replace(0, np.nan) để tránh chia cho 0, sau đó điền lại bằng 0
    result_df["AvgOrderValue"] = (
        result_df["Monetary"] / result_df["Frequency"].replace(0, np.nan)
    ).fillna(0).round(2)

    # Đặc trưng 2: Tenure - số ngày từ lần mua đầu tiên đến snapshot
    first_purchase = working_df.groupby(customer_id_col)["PurchaseDate"].min()
    tenure_days = (snapshot_timestamp - first_purchase).dt.days
    result_df["Tenure"] = result_df[customer_id_col].map(tenure_days).fillna(0).astype(int)

    # Đặc trưng 3: khoảng cách trung bình giữa các lần mua của riêng từng khách
    def average_gap_of_customer(purchase_dates: pd.Series) -> float:
        unique_dates = sorted(purchase_dates.unique())
        if len(unique_dates) < 2:
            # Khách chỉ mua 1 lần: không có khoảng cách nào để tính.
            # Dùng NaN ở đây rồi điền sau, KHÔNG dùng 0 vì 0 mang nghĩa
            # "mua liên tục mỗi ngày" - sai lệch hoàn toàn về mặt ngữ nghĩa.
            return np.nan
        date_series = pd.Series(pd.to_datetime(list(unique_dates)))
        return date_series.diff().dropna().dt.days.mean()

    avg_gap = working_df.groupby(customer_id_col)["PurchaseDate"].apply(average_gap_of_customer)
    result_df["AvgGapDays"] = result_df[customer_id_col].map(avg_gap)

    # Khách mua 1 lần: điền bằng chính Tenure của họ - diễn giải là "đã im
    # lặng suốt từ lần mua đầu đến giờ", hợp lý hơn nhiều so với điền 0.
    result_df["AvgGapDays"] = result_df["AvgGapDays"].fillna(result_df["Tenure"]).round(1)

    print(f"[add_churn_features] Đã bổ sung 3 đặc trưng: "
          f"AvgOrderValue, Tenure, AvgGapDays cho {len(result_df):,} khách hàng.")

    return result_df


# ==============================================================================
# KHỐI TEST NHANH - chỉ chạy khi gọi trực tiếp "python label_generator.py"
# ==============================================================================
if __name__ == "__main__":
    print("TEST NHANH MODULE label_generator.py (dùng dữ liệu giả lập)\n")

    # Dữ liệu giả lập: 3 khách hàng với hành vi khác nhau rõ rệt
    #  - KH 1: mua đều đặn mỗi 30 ngày, CÓ quay lại sau snapshot -> churn = 0
    #  - KH 2: mua vài lần rồi ngừng hẳn trước snapshot           -> churn = 1
    #  - KH 3: chỉ mua đúng 1 lần, không quay lại                 -> churn = 1
    test_transactions = pd.DataFrame({
        "Customer ID": [1, 1, 1, 1, 2, 2, 2, 3],
        "Invoice": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1"],
        "InvoiceDate": pd.to_datetime([
            "2011-01-01", "2011-01-31", "2011-03-02", "2011-06-15",  # KH1
            "2011-01-10", "2011-02-10", "2011-03-12",                 # KH2
            "2011-02-01",                                              # KH3
        ]),
        "TotalPrice": [100, 120, 110, 130, 200, 180, 220, 50],
    })

    print("Dữ liệu giả lập:")
    print(test_transactions)

    print("\n[Test 1] compute_inter_purchase_gaps()")
    gaps = compute_inter_purchase_gaps(test_transactions)
    print(f"Các khoảng cách tính được: {sorted(gaps.tolist())}")
    # KH1 có 3 khoảng: 30, 30, 105 ngày; KH2 có 2 khoảng: 31, 30 ngày
    assert len(gaps) == 5, f"Phải có đúng 5 khoảng cách, nhận được {len(gaps)}!"
    print("✔ Tính đúng số lượng khoảng cách (KH mua 1 lần không đóng góp giá trị nào).")

    print("\n[Test 2] summarize_gap_distribution()")
    gap_summary = summarize_gap_distribution(gaps, percentiles=[50, 75, 90])
    print(gap_summary.to_string(index=False))
    print("✔ Bảng phân phối lập đúng.")

    print("\n[Test 3] generate_churn_labels() - snapshot 2011-04-01, cửa sổ 90 ngày")
    # Cửa sổ quan sát: (2011-04-01, 2011-06-30]
    #  - KH1 mua ngày 2011-06-15 -> nằm trong cửa sổ -> churn = 0
    #  - KH2 lần cuối 2011-03-12 -> không quay lại   -> churn = 1
    #  - KH3 lần cuối 2011-02-01 -> không quay lại   -> churn = 1
    labels = generate_churn_labels(test_transactions,
                                    snapshot_date="2011-04-01",
                                    window_days=90)
    print(labels.to_string(index=False))
    expected = {1: 0, 2: 1, 3: 1}
    for _, row in labels.iterrows():
        customer_id = int(row["Customer ID"])
        assert row["Churn"] == expected[customer_id], \
            f"Khách {customer_id} phải có Churn={expected[customer_id]}!"
    print("✔ Nhãn churn sinh đúng cho cả 3 trường hợp.")

    print("\n[Test 4] build_time_based_snapshots() - kiểm tra 2 cửa sổ không chồng lấn")
    train_snapshot, test_snapshot = build_time_based_snapshots(test_transactions, window_days=30)
    assert train_snapshot < test_snapshot, "Mốc train phải sớm hơn mốc test!"
    print("✔ Hai mốc snapshot tách đúng theo thứ tự thời gian.")

    print("\n[Test 5] add_churn_features() - kiểm tra chống rò rỉ dữ liệu")
    from src.rfm.rfm_calculator import calculate_rfm
    rfm_at_snapshot = calculate_rfm(test_transactions, snapshot_date="2011-04-01")
    features = add_churn_features(rfm_at_snapshot, test_transactions,
                                   snapshot_date="2011-04-01")
    print(features.to_string(index=False))
    # KH1 tính đến 2011-04-01 chỉ có 3 hoá đơn (giao dịch 2011-06-15 nằm SAU
    # snapshot nên phải bị loại) -> Frequency = 3, không phải 4
    kh1_frequency = features.loc[features["Customer ID"] == 1, "Frequency"].iloc[0]
    assert kh1_frequency == 3, \
        f"Khách 1 phải có Frequency=3 tại snapshot (giao dịch sau snapshot bị loại), " \
        f"nhận được {kh1_frequency}!"
    assert "AvgOrderValue" in features.columns and "Tenure" in features.columns \
        and "AvgGapDays" in features.columns, "Thiếu đặc trưng phái sinh!"
    print("✔ Đặc trưng bổ sung đúng, và giao dịch SAU snapshot đã bị loại "
          "(cơ chế chống rò rỉ dữ liệu hoạt động).")

    print("\n✔ TEST THÀNH CÔNG toàn bộ module label_generator.py")
