"""
Trang Đánh giá mô hình Churn gồm 3 tab:
  1. So sánh mô hình (bảng và biểu đồ từ Churn_Model_Results.csv, có baseline).
  2. ROC và ngưỡng quyết định (tính trực tiếp từ các file .pkl trên tập Test).
  3. Phương pháp và hạn chế (tự đánh giá trung thực).
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from Common import (
    MAU_MO_HINH, MO_HINH_MAC_DINH, THONG_TIN_DU_AN, dinh_dang_bieu_do, hien_anh, hien_bang,
    hien_loi_thieu_tep, hien_tieu_de_trang, mau_mo_hinh, tai_du_doan, tai_du_lieu_test,
    tai_ket_qua_mo_hinh, tinh_baseline_f1, tinh_chi_so_theo_nguong, tinh_roc,
    tinh_xac_suat_test,
)

CAC_CHI_SO = ["Precision", "Recall", "F1", "AUC"]


def _thu_tu_mo_hinh(ten_mo_hinh: str):
    ten_goc = ten_mo_hinh.replace(" (Tuned)", "")
    thu_tu_goc = list(MAU_MO_HINH).index(ten_goc) if ten_goc in MAU_MO_HINH else 99
    return (thu_tu_goc, "(Tuned)" in ten_mo_hinh)


def _tab_so_sanh(ket_qua, du_doan) -> None:
    ti_le_churn = float(du_doan["Churn thực tế"].mean())
    f1_baseline = tinh_baseline_f1(ti_le_churn)
    tot_nhat_f1 = ket_qua.loc[ket_qua["F1"].idxmax()]
    tot_nhat_auc = ket_qua.loc[ket_qua["AUC"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1 cao nhất", f"{tot_nhat_f1['F1']:.4f}", delta=str(tot_nhat_f1["Model name"]), delta_color="off")
    c2.metric("AUC cao nhất", f"{tot_nhat_auc['AUC']:.4f}", delta=str(tot_nhat_auc["Model name"]), delta_color="off")
    c3.metric("Baseline F1", f"{f1_baseline:.4f}", help="Mô hình Baseline luôn đoán 'rời bỏ' (Precision = Tỷ lệ Churn, Recall = 1)")
    c4.metric("Tỷ lệ Churn tập Test", f"{ti_le_churn * 100:.1f}%")

    # Bảng kết quả theo thứ tự nhóm thuật toán
    thu_tu = sorted(ket_qua["Model name"], key=_thu_tu_mo_hinh)
    ket_qua_sap_xep = ket_qua.set_index("Model name").loc[thu_tu].reset_index()
    hien_bang(ket_qua_sap_xep.round(4))

    # Biểu đồ cột nhóm: bản gốc tô đặc, bản Tuned tô sọc cùng màu
    bieu_do = go.Figure()
    for ten in thu_tu:
        dong = ket_qua_sap_xep[ket_qua_sap_xep["Model name"] == ten].iloc[0]
        la_tuned = "(Tuned)" in ten
        bieu_do.add_trace(go.Bar(
            name=ten, x=CAC_CHI_SO, y=[float(dong[c]) for c in CAC_CHI_SO],
            marker=dict(color=mau_mo_hinh(ten),
                        pattern=dict(shape="/" if la_tuned else "", fgcolor="white")),
            text=[f"{float(dong[c]):.3f}" for c in CAC_CHI_SO], textposition="outside",
        ))
    bieu_do.update_layout(barmode="group", title="So sánh các mô hình: Tham số mặc định (đặc) và Sau GridSearchCV (sọc)",
                          yaxis=dict(range=[0, 1.1], title="Giá trị"), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(dinh_dang_bieu_do(bieu_do, 440), key="dg_so_sanh")

    # Nhận xét tự sinh từ số liệu, không viết cứng
    xep_hang = ket_qua.sort_values("F1", ascending=False)
    chenh_f1 = float(xep_hang["F1"].iloc[0] - xep_hang["F1"].iloc[1]) if len(xep_hang) > 1 else 0.0
    st.markdown(
        f"""
        <div class="iq-card">
            <div class="section-title">📝 Nhận xét</div>
            <div class="muted">
                • <b>{tot_nhat_f1['Model name']}</b> dẫn đầu F1 ({tot_nhat_f1['F1']:.4f}), hơn Baseline
                <b>{tot_nhat_f1['F1'] - f1_baseline:+.4f}</b>; mô hình có AUC cao nhất là
                <b>{tot_nhat_auc['Model name']}</b> ({tot_nhat_auc['AUC']:.4f}).<br>
                • Chênh lệch F1 giữa hai mô hình đứng đầu chỉ <b>{chenh_f1:.4f}</b>, chưa được kiểm định thống kê
                nên thứ hạng không mang tính quyết định.<br>
                • Tinh chỉnh siêu tham số không phải lúc nào cũng cải thiện: Hãy so sánh từng cặp gốc và Tuned trong bảng.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Biểu đồ từ Notebook (Benchmark và độ quan trọng đặc trưng)"):
        co_hinh = hien_anh("Benchmark", "Benchmark Apriori và FP-Growth")
        co_hinh = hien_anh("Feature Importance", "Độ quan trọng đặc trưng của Random Forest") or co_hinh
        if not co_hinh:
            st.info("Chưa tìm thấy ảnh trong thư mục Figures.")


def _tab_roc_nguong() -> None:
    xac_suat_cac_mo_hinh = tinh_xac_suat_test()
    if not xac_suat_cac_mo_hinh:
        st.info("Không tìm thấy mô hình .pkl nên không tính được ROC. Hãy chạy Notebook 05.")
        return
    _, y_that = tai_du_lieu_test()
    f1_baseline = tinh_baseline_f1(float(y_that.mean()))

    st.markdown(
        '<div class="muted">Các đường ROC bên dưới được tính trực tiếp từ các File mô hình <b>.pkl</b> '
        "trên tập Test, nên giá trị AUC phải khớp với bảng ở tab trước (với mô hình không tinh chỉnh).</div>",
        unsafe_allow_html=True,
    )

    cot_chon, cot_nguong = st.columns(2)
    with cot_chon:
        ds_mo_hinh = list(xac_suat_cac_mo_hinh)
        chi_so = ds_mo_hinh.index(MO_HINH_MAC_DINH) if MO_HINH_MAC_DINH in ds_mo_hinh else 0
        ten_chon = st.selectbox("Mô hình", ds_mo_hinh, index=chi_so, key="dg_mo_hinh")
    with cot_nguong:
        nguong = st.slider("Ngưỡng quyết định (Xác suất ≥ ngưỡng thì dự đoán 'rời bỏ')", 0.05, 0.95, 0.5, 0.05, key="dg_nguong")

    chi_so_nguong = tinh_chi_so_theo_nguong(y_that, xac_suat_cac_mo_hinh[ten_chon], nguong)

    cot_roc, cot_nham = st.columns(2, gap="large")
    with cot_roc:
        bieu_do_roc = go.Figure()
        for ten, xac_suat in xac_suat_cac_mo_hinh.items():
            fpr, tpr, auc = tinh_roc(y_that, xac_suat)
            bieu_do_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{ten} (AUC = {auc:.4f})",
                                             line=dict(color=mau_mo_hinh(ten), width=2.5)))
        bieu_do_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Đoán ngẫu nhiên",
                                         line=dict(color="gray", dash="dash")))
        # Điểm hoạt động của mô hình đang chọn tại ngưỡng hiện tại
        fpr_hien_tai = chi_so_nguong["fp"] / max(chi_so_nguong["fp"] + chi_so_nguong["tn"], 1)
        bieu_do_roc.add_trace(go.Scatter(x=[fpr_hien_tai], y=[chi_so_nguong["recall"]], mode="markers",
                                         name=f"{ten_chon} tại ngưỡng {nguong:.2f}",
                                         marker=dict(size=13, color="#111827", symbol="diamond")))
        bieu_do_roc.update_layout(title="Đường cong ROC trên tập Test", xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(dinh_dang_bieu_do(bieu_do_roc, 420), key="dg_roc")

    with cot_nham:
        ma_tran = [[chi_so_nguong["tn"], chi_so_nguong["fp"]], [chi_so_nguong["fn"], chi_so_nguong["tp"]]]
        bieu_do_nham = px.imshow(
            ma_tran, text_auto=True, color_continuous_scale="Blues",
            x=["Dự đoán: Ở lại", "Dự đoán: Rời bỏ"], y=["Thực tế: Ở lại", "Thực tế: Rời bỏ"],
            title=f"Ma trận nhầm lẫn - {ten_chon} (ngưỡng {nguong:.2f})",
        )
        bieu_do_nham.update_coloraxes(showscale=False)
        st.plotly_chart(dinh_dang_bieu_do(bieu_do_nham, 420), key="dg_nham")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{chi_so_nguong['precision']:.4f}")
    m2.metric("Recall", f"{chi_so_nguong['recall']:.4f}")
    m3.metric("F1", f"{chi_so_nguong['f1']:.4f}", delta=f"{chi_so_nguong['f1'] - f1_baseline:+.4f} so với baseline")
    m4.metric("Accuracy", f"{chi_so_nguong['accuracy']:.4f}")

    # Phân phối xác suất: cho thấy mô hình có bị 'bão hoà' về 0 hoặc 1 không
    du_lieu_hist = {"Xác suất Churn": xac_suat_cac_mo_hinh[ten_chon],
                    "Thực tế": ["Rời bỏ" if y == 1 else "Ở lại" for y in y_that]}
    bieu_do_hist = px.histogram(du_lieu_hist, x="Xác suất Churn", color="Thực tế", nbins=40, barmode="overlay",
                                opacity=0.7, color_discrete_map={"Rời bỏ": "#ef4444", "Ở lại": "#22c55e"},
                                title=f"Phân phối xác suất dự đoán - {ten_chon}")
    st.plotly_chart(dinh_dang_bieu_do(bieu_do_hist, 320), key="dg_hist")
    ty_le_bao_hoa = float(((xac_suat_cac_mo_hinh[ten_chon] < 0.05) | (xac_suat_cac_mo_hinh[ten_chon] > 0.95)).mean())
    st.caption(f"{ty_le_bao_hoa * 100:.1f}% khách có xác suất sát 0 hoặc 1. Tỷ lệ này càng cao thì mô hình càng 'cực đoan' "
               "(thường gặp ở Naive Bayes), và việc đổi ngưỡng càng ít tác dụng.")


# =============================================================================
# TAB 3: PHƯƠNG PHÁP VÀ HẠN CHẾ
# =============================================================================
def _tab_phuong_phap(du_doan) -> None:
    thong_tin = THONG_TIN_DU_AN
    ti_le_test = float(du_doan["Churn thực tế"].mean()) * 100
    st.markdown(
        f"""
        <div class="iq-card">
            <div class="section-title">🧪 Thiết kế thực nghiệm</div>
            <div class="muted">
                • <b>Nhãn Churn:</b> khách đã có giao dịch tính đến mốc Snapshot nhưng không mua lại trong
                <b>{thong_tin['cua_so_churn_ngay']}</b> ngày kế tiếp. Cửa sổ lấy theo phân vị 75 của khoảng cách giữa các lần mua,
                tính chỉ trên dữ liệu trước mốc Train.<br>
                • <b>Tách Train/Test theo thời gian:</b> Train tại Snapshot {thong_tin['snapshot_train']}
                ({thong_tin['so_khach_train']:,} khách, Churn {thong_tin['ti_le_churn_train']}%);
                Test tại Snapshot {thong_tin['snapshot_test']} ({len(du_doan):,} khách, Churn {ti_le_test:.1f}%).<br>
                • <b>Chống rò rỉ:</b> mọi đặc trưng chỉ dùng giao dịch trước Snapshot; SMOTE chỉ áp trên Train.<br>
                • <b>Thước đo:</b> Precision, Recall, F1, AUC và so sánh với baseline (không dùng mỗi Accuracy vì lớp mất cân bằng).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="iq-card">
            <div class="section-title">⚠️ Hạn chế đã tự đánh giá</div>
            <div class="muted">
                • <b>Tinh chỉnh siêu tham số chưa tách SMOTE khỏi vòng CV:</b> GridSearchCV chạy trên tập Train đã SMOTE
                với fold ngẫu nhiên nên điểm CV có thể lạc quan. Đánh giá cuối cùng trên tập Test tách theo thời gian
                không bị ảnh hưởng. Hướng cải tiến: đặt SMOTE trong pipeline của CV.<br>
                • <b>Các mô hình dẫn đầu chênh nhau rất ít</b> và chưa được kiểm định thống kê; mức vượt baseline F1 chỉ vài điểm phần trăm.
                Mô hình dùng để xuất dự đoán được chọn theo F1 trên chính tập Test.<br>
                • <b>Xác suất của Naive Bayes không được hiệu chuẩn</b>, chỉ nên dùng để xếp hạng rủi ro.<br>
                • <b>Cửa sổ Churn dùng chung cho mọi khách</b>; khách chỉ mua một lần không tạo khoảng cách giữa các lần mua;
                nhãn Test kết thúc đúng ngày cuối của dữ liệu.<br>
                • <b>Dữ liệu tập trung ở Vương quốc Anh</b> (khoảng 92% số dòng), khó khái quát ra thị trường khác.<br>
                • <b>Luật kết hợp chỉ thể hiện sự đồng xuất hiện</b>; luật Lift cao nhất phần lớn là mua theo bộ sưu tập;
                luật theo phân khúc loại các hoá đơn thiếu Customer ID.<br>
                • <b>Phân khúc cho khách nhập mới</b> dùng K-Means toàn kỳ của Notebook 04, có thể khác với phân khúc
                tính tại Snapshot Test ở Notebook 06.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def Evaluation() -> None:
    hien_tieu_de_trang("📊", "Đánh giá mô hình Churn",
                       "So sánh Decision Tree, Random Forest, Naive Bayes trên tập Test tách theo thời gian - Notebook 05")
    try:
        ket_qua = tai_ket_qua_mo_hinh()
        du_doan = tai_du_doan()
    except FileNotFoundError as loi:
        hien_loi_thieu_tep(loi)
        return

    tab_so_sanh, tab_roc, tab_phuong_phap = st.tabs(["So sánh mô hình", "ROC và ngưỡng quyết định", "Phương pháp và hạn chế"])
    with tab_so_sanh:
        _tab_so_sanh(ket_qua, du_doan)
    with tab_roc:
        _tab_roc_nguong()
    with tab_phuong_phap:
        _tab_phuong_phap(du_doan)
