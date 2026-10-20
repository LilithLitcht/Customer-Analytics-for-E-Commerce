import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


THU_MUC_DEMO = Path(__file__).resolve().parent
GOC_DU_AN = THU_MUC_DEMO.parent
THU_MUC_INTERIM = GOC_DU_AN / "Data" / "Interim"
THU_MUC_MODELS = GOC_DU_AN / "Models"
THU_MUC_FIGURES = GOC_DU_AN / "Figures"

# Thứ tự đặc trưng đúng như lúc huấn luyện ở Notebook 05 (mô hình .pkl không lưu tên cột)
DAC_TRUNG = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "Tenure", "AvgGapDays"]

# Tên cột trong file Customer_Churn_Predictions.csv
DOI_TEN_COT_FILE = {"Avg Order Value": "AvgOrderValue", "Avg Gap Days": "AvgGapDays"}

# Mô tả ngắn từng đặc trưng
MO_TA_DAC_TRUNG = {
    "Recency": "Số ngày kể từ lần mua gần nhất đến thời điểm dự đoán",
    "Frequency": "Số hoá đơn đã mua",
    "Monetary": "Tổng giá trị đã chi tiêu (GBP)",
    "AvgOrderValue": "Giá trị trung bình mỗi hoá đơn = Monetary / Frequency",
    "Tenure": "Số ngày kể từ lần mua đầu tiên đến thời điểm dự đoán",
    "AvgGapDays": "Khoảng cách trung bình (ngày) giữa các lần mua",
}

# Tên mô hình
TEP_MO_HINH = {
    "Naive Bayes": "Naive_Bayes_Churn.pkl",
    "Random Forest": "Random_Forest_Churn.pkl",
    "Decision Tree": "Decision_Tree_Churn.pkl",
}

MO_HINH_MAC_DINH = "Naive Bayes" 
TEP_KMEANS = "KMeans_RFM.pkl"

# Màu từng thuật toán 
MAU_MO_HINH = {
    "Decision Tree": "#2E86AB",
    "Random Forest": "#E67E22",
    "Naive Bayes": "#C0392B",
}

# Ngưỡng chia mức rủi ro để HIỂN THỊ
# Quyết định phân lớp của mô hình vẫn dùng ngưỡng 0,5.
NGUONG_RUI_RO_THAP = 0.40
NGUONG_RUI_RO_CAO = 0.70

# Thông tin tổng hợp lấy từ báo cáo/notebook. Nếu chạy lại pipeline và số liệu đổi, chỉ cần sửa tại đây.
THONG_TIN_DU_AN = {
    "so_dong_goc": 1_067_371,          # Số dòng dữ liệu thô 
    "so_dong_sau_lam_sach": 1_006_013,  # Số dòng sau làm sạch 
    "ngay_cuoi_du_lieu": "09/12/2011",
    "cua_so_churn_ngay": 67,           # Cửa sổ Churn = P75 inter-purchase gap trên dữ liệu trước mốc Train
    "snapshot_train": "28/07/2011",
    "snapshot_test": "03/10/2011",
    "so_khach_train": 5100,
    "ti_le_churn_train": 70.1,         # Đơn vị %
}

# Kịch bản mẫu để thử nhanh phần nhập khách hàng mới
KICH_BAN_MAU = {
    "Khách VIP (Mua nhiều, Mua gần đây)": dict(Recency=20, Frequency=15, Monetary=8000.0, Tenure=600, AvgGapDays=35.0),
    "Khách phổ thông (Mua thỉnh thoảng)": dict(Recency=120, Frequency=3, Monetary=900.0, Tenure=300, AvgGapDays=100.0),
    "Khách lâu không quay lại": dict(Recency=420, Frequency=2, Monetary=450.0, Tenure=500, AvgGapDays=80.0),
    "Khách mới mua lần đầu": dict(Recency=10, Frequency=1, Monetary=120.0, Tenure=10, AvgGapDays=10.0),
}


def mau_phan_khuc(ten_phan_khuc: str) -> str:
    ten = str(ten_phan_khuc).lower()
    if "vip" in ten:
        return "#22c55e"                    # Rủi ro thấp
    if "nguy cơ" in ten or "rời bỏ" in ten:
        return "#ef4444"                    # Rủi ro cao
    if "trung thành" in ten:
        return "#f59e0b"                    # Trung bình
    return "#0ea5e9"


def icon_phan_khuc(ten_phan_khuc: str) -> str:
    ten = str(ten_phan_khuc).lower()
    if "vip" in ten:
        return "💎"
    if "nguy cơ" in ten or "rời bỏ" in ten:
        return "⚠️"
    if "trung thành" in ten:
        return "🤝"
    return "👤"


def ban_do_mau_phan_khuc(danh_sach_ten) -> dict:
    return {ten: mau_phan_khuc(ten) for ten in danh_sach_ten}


def mau_mo_hinh(ten_mo_hinh: str) -> str:
    ten_goc = str(ten_mo_hinh).replace(" (Tuned)", "").strip()
    return MAU_MO_HINH.get(ten_goc, "#0ea5e9")

def hien_tieu_de_trang(icon: str, tieu_de: str, mo_ta: str) -> None:
    st.markdown(
        f"""
        <div class="sky-card fade-in" style="border-left:5px solid #0ea5e9;">
            <div style="display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;">
                <div style="font-size:2.2rem;">{icon}</div>
                <div>
                    <div style="font-size:1.65rem;font-weight:800;color:#0369a1;">{tieu_de}</div>
                    <div class="muted">{mo_ta}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hien_bang(du_lieu, **tuy_chon) -> None:
    try:
        st.dataframe(du_lieu, hide_index=True, width="stretch", **tuy_chon)
    except Exception:
        try:
            st.dataframe(du_lieu, hide_index=True, use_container_width=True, **tuy_chon)
        except Exception:
            st.dataframe(du_lieu, hide_index=True, **tuy_chon)


def hien_anh(tu_khoa: str, chu_thich: str = None) -> bool:
    if not THU_MUC_FIGURES.exists():
        return False
    for duong_dan in sorted(THU_MUC_FIGURES.glob("*.png")):
        if tu_khoa.lower() in duong_dan.name.lower():
            st.image(str(duong_dan), caption=chu_thich or duong_dan.stem)
            return True
    return False


def hien_loi_thieu_tep(loi: Exception) -> None:
    st.error(f"Không tải được dữ liệu cần thiết: {loi}")
    st.info(
        "Hãy chạy lại các Notebook 01 → 06 để sinh đủ File trong `Data/Interim` và `Models`, "
        "rồi tải lại trang."
    )


def dinh_dang_bieu_do(hinh: go.Figure, chieu_cao: int = 360) -> go.Figure:
    hinh.update_layout(
        height=chieu_cao,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.8)",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return hinh


def _doc_csv(ten_tep: str) -> pd.DataFrame:
    duong_dan = THU_MUC_INTERIM / ten_tep
    if not duong_dan.exists():
        raise FileNotFoundError(f"Không tìm thấy File {duong_dan}")
    return pd.read_csv(duong_dan, encoding="utf-8-sig")


def _chuan_hoa_ma_khach(df: pd.DataFrame) -> pd.DataFrame:
    if "Customer ID" in df.columns:
        df["Customer ID"] = df["Customer ID"].astype(float).astype(int)
    return df


@st.cache_data(show_spinner="Đang tải dữ liệu phân khúc...")
def tai_phan_khuc() -> pd.DataFrame:
    return _chuan_hoa_ma_khach(_doc_csv("Customer_Segments_Result.csv"))


@st.cache_data(show_spinner="Đang tải kết quả mô hình...")
def tai_ket_qua_mo_hinh() -> pd.DataFrame:
    return _doc_csv("Churn_Model_Results.csv")


@st.cache_data(show_spinner="Đang tải dự đoán Churn...")
def tai_du_doan() -> pd.DataFrame:
    df = _chuan_hoa_ma_khach(_doc_csv("Customer_Churn_Predictions.csv"))
    df = df.rename(columns=DOI_TEN_COT_FILE)

    phan_khuc = tai_phan_khuc()[["Customer ID", "Tên phân khúc"]]
    df = df.merge(phan_khuc, on="Customer ID", how="left")
    df["Tên phân khúc"] = df["Tên phân khúc"].fillna("Không xác định")
    return df


def chuan_hoa_cot_luat(df: pd.DataFrame) -> pd.DataFrame:
    bang_ten = {
        "antecedents": "Antecedents", "antecedentsname": "Antecedents Name",
        "consequents": "Consequents", "consequentsname": "Consequents Name",
        "support": "Support", "confidence": "Confidence",
        "lift": "Lift", "kulczynski": "Kulczynski",
    }
    doi_ten = {}
    for cot in df.columns:
        khoa = re.sub(r"[\s_]+", "", str(cot)).lower()
        if khoa in bang_ten:
            doi_ten[cot] = bang_ten[khoa]
    return df.rename(columns=doi_ten)


def _them_cot_phu_tro_cho_luat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for cot in ("Antecedents", "Consequents", "Antecedents Name", "Consequents Name"):
        df[cot] = df[cot].astype(str).str.strip()
    # Mã sản phẩm không chứa dấu phẩy nên đếm dấu phẩy để biết số sản phẩm mỗi vế
    df["So SP ve trai"] = df["Antecedents"].str.count(",") + 1
    df["So SP ve phai"] = df["Consequents"].str.count(",") + 1
    df["Khoa cap"] = ["|".join(sorted((a, c))) for a, c in zip(df["Antecedents"], df["Consequents"])]
    return df


@st.cache_data(show_spinner="Đang tải luật kết hợp...")
def tai_luat_tong() -> pd.DataFrame:
    return _them_cot_phu_tro_cho_luat(chuan_hoa_cot_luat(_doc_csv("Association_Rules_Result.csv")))


@st.cache_data(show_spinner="Đang tải luật kết hợp theo phân khúc...")
def tai_luat_phan_khuc() -> pd.DataFrame:
    df = chuan_hoa_cot_luat(_doc_csv("Association_Rules_By_Segment.csv"))
    return _them_cot_phu_tro_cho_luat(df)


@st.cache_resource(show_spinner="Đang tải các mô hình Churn...")
def tai_mo_hinh() -> dict:
    ket_qua = {}
    for ten, tep in TEP_MO_HINH.items():
        duong_dan = THU_MUC_MODELS / tep
        if duong_dan.exists():
            ket_qua[ten] = joblib.load(duong_dan)
    return ket_qua


def _bien_doi_rfm(df_rfm: pd.DataFrame) -> pd.DataFrame:
    ket_qua = df_rfm[["Recency", "Frequency", "Monetary"]].astype(float).copy()
    for cot in ("Frequency", "Monetary"):
        ket_qua[cot] = np.log1p(ket_qua[cot].clip(lower=0))
    return ket_qua


@st.cache_resource(show_spinner="Đang dựng bộ phân khúc RFM...")
def tai_bo_phan_khuc():
    duong_dan = THU_MUC_MODELS / TEP_KMEANS
    if not duong_dan.exists():
        return None

    kmeans = joblib.load(duong_dan)
    df = tai_phan_khuc()
    du_lieu_bien_doi = _bien_doi_rfm(df)
    bo_chuan_hoa = StandardScaler().fit(du_lieu_bien_doi)
    cum_du_doan = kmeans.predict(bo_chuan_hoa.transform(du_lieu_bien_doi))

    bang = pd.DataFrame({"cum": cum_du_doan, "ten": df["Tên phân khúc"].to_numpy()})
    ten_cum = bang.groupby("cum")["ten"].agg(lambda s: s.mode().iloc[0]).to_dict()
    do_khop = float((bang["cum"].map(ten_cum) == bang["ten"]).mean())

    return {"kmeans": kmeans, "bo_chuan_hoa": bo_chuan_hoa, "ten_cum": ten_cum, "do_khop": do_khop}


def xac_dinh_phan_khuc(recency: float, frequency: float, monetary: float):
    bo = tai_bo_phan_khuc()
    if bo is None:
        return None
    mau = pd.DataFrame([{"Recency": recency, "Frequency": frequency, "Monetary": monetary}])
    cum = int(bo["kmeans"].predict(bo["bo_chuan_hoa"].transform(_bien_doi_rfm(mau)))[0])
    return bo["ten_cum"].get(cum)


def phan_loai_rui_ro(xac_suat: float) -> dict:
    if xac_suat < NGUONG_RUI_RO_THAP:
        return {"nhan": "Thấp", "icon": "🟢", "mau": "#22c55e", "nen": "#f0fdf4",
                "mo_ta": "Khả năng quay lại mua cao"}
    if xac_suat < NGUONG_RUI_RO_CAO:
        return {"nhan": "Trung bình", "icon": "🟡", "mau": "#f59e0b", "nen": "#fffbeb",
                "mo_ta": "Cần theo dõi thêm"}
    return {"nhan": "Cao", "icon": "🔴", "mau": "#ef4444", "nen": "#fef2f2",
            "mo_ta": "Nguy cơ rời bỏ cao, cần can thiệp"}


def lay_hanh_dong(ten_phan_khuc: str, xac_suat: float) -> str:
    ten = str(ten_phan_khuc).lower()
    canh_bao = None
    if "vip" in ten:
        co_ban = "Chương trình khách hàng thân thiết, ưu đãi độc quyền, chăm sóc và cá nhân hoá trải nghiệm."
        canh_bao = "Đây là khách giá trị cao nhưng rủi ro rời bỏ lớn: Ưu tiên liên hệ chăm sóc trực tiếp ngay."
    elif "nguy cơ" in ten or "rời bỏ" in ten:
        co_ban = ("Chiến dịch tái kích hoạt: Email nhắc nhở, ưu đãi có thời hạn "
                  "và thông điệp cá nhân hoá theo lịch sử mua hàng.")
    elif "trung thành" in ten:
        co_ban = ("Gợi ý mua kèm theo các luật kết hợp riêng của phân khúc (xem bên dưới); "
                  "nên thử nghiệm quy mô nhỏ trước khi triển khai rộng.")
        canh_bao = "Rủi ro rời bỏ cao: Kết hợp ưu đãi giữ chân ngay trong lần mua kế tiếp."
    else:
        co_ban = "Theo dõi hành vi mua và chăm sóc theo nhu cầu."
    if xac_suat >= NGUONG_RUI_RO_CAO and canh_bao:
        return f"{co_ban} {canh_bao}"
    return co_ban


def ve_dong_ho_rui_ro(xac_suat: float) -> go.Figure:
    mau = phan_loai_rui_ro(xac_suat)["mau"]
    thap, cao = NGUONG_RUI_RO_THAP * 100, NGUONG_RUI_RO_CAO * 100
    hinh = go.Figure(go.Indicator(
        mode="gauge+number",
        value=xac_suat * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"color": mau, "size": 30}},
        title={"text": "Xác suất Churn của mô hình", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": mau, "thickness": 0.28},
            "steps": [
                {"range": [0, thap], "color": "#dcfce7"},
                {"range": [thap, cao], "color": "#fef9c3"},
                {"range": [cao, 100], "color": "#fee2e2"},
            ],
        },
    ))
    hinh.update_layout(height=260, margin=dict(t=60, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    return hinh


def loc_luat(df_luat: pd.DataFrame, chi_1_1: bool = True, lift_toi_thieu: float = 0.0,
             kulczynski_toi_thieu: float = 0.0, support_toi_thieu: float = 0.0,
             confidence_toi_thieu: float = 0.0, bo_trung_chieu: bool = True) -> pd.DataFrame:
    ket_qua = df_luat
    if chi_1_1:
        ket_qua = ket_qua[(ket_qua["So SP ve trai"] == 1) & (ket_qua["So SP ve phai"] == 1)]
    ket_qua = ket_qua[
        (ket_qua["Lift"] >= lift_toi_thieu)
        & (ket_qua["Kulczynski"] >= kulczynski_toi_thieu)
        & (ket_qua["Support"] >= support_toi_thieu)
        & (ket_qua["Confidence"] >= confidence_toi_thieu)
    ]
    ket_qua = ket_qua.sort_values(["Lift", "Confidence"], ascending=[False, False])
    if bo_trung_chieu:
        ket_qua = ket_qua.drop_duplicates("Khoa cap")
    return ket_qua


def dinh_dang_bang_luat(df_luat: pd.DataFrame) -> pd.DataFrame:
    bang = df_luat[["Antecedents Name", "Consequents Name", "Support", "Confidence", "Lift", "Kulczynski"]].copy()
    bang.columns = ["Khách mua", "Thường mua kèm", "Support", "Confidence", "Lift", "Kulczynski"]
    return bang.round({"Support": 4, "Confidence": 3, "Lift": 2, "Kulczynski": 3}).reset_index(drop=True)


def lay_goi_y_phan_khuc(ten_phan_khuc: str, so_luat: int = 3) -> pd.DataFrame:
    try:
        luat = tai_luat_phan_khuc()
    except FileNotFoundError:
        return pd.DataFrame()
    luat = luat[luat["Tên phân khúc"] == ten_phan_khuc]
    return loc_luat(luat).head(so_luat)


def tinh_baseline_f1(ti_le_churn: float) -> float:
    return 2 * ti_le_churn / (1 + ti_le_churn)


@st.cache_data(show_spinner="Đang chuẩn bị tập Test...")
def tai_du_lieu_test():
    df = tai_du_doan()
    return df[DAC_TRUNG].astype(float).reset_index(drop=True), df["Churn thực tế"].astype(int).to_numpy()


@st.cache_data(show_spinner="Đang tính xác suất dự đoán trên tập Test...")
def tinh_xac_suat_test() -> dict:
    x_test, _ = tai_du_lieu_test()
    return {ten: mo_hinh.predict_proba(x_test)[:, 1] for ten, mo_hinh in tai_mo_hinh().items()}


def tinh_chi_so_theo_nguong(y_that: np.ndarray, xac_suat: np.ndarray, nguong: float) -> dict:
    du_doan = (xac_suat >= nguong).astype(int)
    tp = int(((du_doan == 1) & (y_that == 1)).sum())
    fp = int(((du_doan == 1) & (y_that == 0)).sum())
    fn = int(((du_doan == 0) & (y_that == 1)).sum())
    tn = int(((du_doan == 0) & (y_that == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / len(y_that), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def tinh_roc(y_that: np.ndarray, xac_suat: np.ndarray):
    fpr, tpr, _ = roc_curve(y_that, xac_suat)
    return fpr, tpr, float(roc_auc_score(y_that, xac_suat))