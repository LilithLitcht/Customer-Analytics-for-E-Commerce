import plotly.graph_objects as go
import streamlit as st

from Common import (
    THONG_TIN_DU_AN, dinh_dang_bieu_do, hien_loi_thieu_tep, hien_tieu_de_trang,
    mau_mo_hinh, mau_phan_khuc, tai_du_doan, tai_ket_qua_mo_hinh, tai_luat_phan_khuc,
    tai_luat_tong, tai_phan_khuc, tinh_baseline_f1,
)


def _the_goc_nhin(icon: str, tieu_de: str, cau_hoi: str, thuat_toan: str, ket_qua: str, mau: str) -> str:
    return f"""
    <div class="sky-card fade-in" style="border-top:4px solid {mau};min-height:230px;">
        <div style="font-size:2rem;">{icon}</div>
        <div style="font-weight:800;font-size:1.1rem;color:{mau};margin:.2rem 0;">{tieu_de}</div>
        <div style="font-weight:600;color:#0f172a;">{cau_hoi}</div>
        <div class="muted" style="font-size:.88rem;margin-top:.4rem;">Thuật toán: {thuat_toan}</div>
        <hr class="fancy-divider" style="margin:10px 0;">
        <div style="font-size:.95rem;color:#0f172a;">{ket_qua}</div>
    </div>
    """


def Home() -> None:
    hien_tieu_de_trang(
        "🛍️", "Customer Analytics for E-Commerce",
        "Dataset: <b>Online Retail II</b> (UCI) - Hiểu và giữ chân khách hàng bằng khai phá dữ liệu",
    )

    # Tải dữ liệu; thiếu file thì báo lỗi và dừng
    try:
        phan_khuc = tai_phan_khuc()
        luat_tong = tai_luat_tong()
        luat_phan_khuc = tai_luat_phan_khuc()
        ket_qua = tai_ket_qua_mo_hinh()
        du_doan = tai_du_doan()
    except FileNotFoundError as loi:
        hien_loi_thieu_tep(loi)
        return

    # Các số liệu dùng nhiều lần
    mo_hinh_tot_nhat = ket_qua.loc[ket_qua["F1"].idxmax()]
    ti_le_churn_test = float(du_doan["Churn thực tế"].mean())
    f1_baseline = tinh_baseline_f1(ti_le_churn_test)
    doanh_thu = phan_khuc.groupby("Tên phân khúc")["Monetary"].sum()
    ten_vip = next((ten for ten in doanh_thu.index if "vip" in ten.lower()), doanh_thu.idxmax())
    ty_trong_vip = doanh_thu[ten_vip] / doanh_thu.sum() * 100

    # Hàng chỉ số tổng quan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Khách hàng được phân tích", f"{len(phan_khuc):,}",
              help="Số khách hàng có Customer ID sau khi làm sạch dữ liệu")
    c2.metric("Phân khúc RFM", f"{phan_khuc['Tên phân khúc'].nunique()}",
              help="Số cụm chọn bằng Silhouette Score ở Notebook 04")
    c3.metric("Luật kết hợp", f"{len(luat_tong):,}",
              help=f"Ngoài ra có {len(luat_phan_khuc):,} luật chạy riêng cho từng phân khúc")
    c4.metric("F1 cao nhất (Churn)", f"{mo_hinh_tot_nhat['F1']:.4f}", delta=str(mo_hinh_tot_nhat["Model name"]),
              delta_color="off", help="Đo trên tập Test tách theo thời gian")
    st.divider()

    # Ba góc nhìn
    st.markdown("#### 🎯 Một bài toán, ba góc nhìn")
    st.markdown(
        '<div class="muted">Cả ba kỹ thuật cùng trả lời câu hỏi: '
        '<b>"Làm sao hiểu và giữ chân khách hàng tốt hơn?"</b></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    g1, g2, g3 = st.columns(3, gap="medium")
    with g1:
        st.markdown(_the_goc_nhin(
            "🛒", "Market Basket Analysis", "Khách mua gì thì thường mua kèm gì?",
            "Apriori, FP-Growth",
            f"<b>{len(luat_tong):,}</b> luật kết hợp, trong đó có "
            f"<b>{len(luat_phan_khuc):,}</b> luật riêng cho từng phân khúc",
            "#0ea5e9"), unsafe_allow_html=True)
    with g2:
        st.markdown(_the_goc_nhin(
            "💎", "RFM Clustering", "Khách hàng nào đáng giá bao nhiêu?",
            "K-Means, K-Medoids (PAM tự cài đặt)",
            f"<b>{phan_khuc['Tên phân khúc'].nunique()}</b> phân khúc; nhóm <b>{ten_vip}</b> "
            f"đóng góp khoảng <b>{ty_trong_vip:.1f}%</b> tổng giá trị chi tiêu",
            "#6d28d9"), unsafe_allow_html=True)
    with g3:
        st.markdown(_the_goc_nhin(
            "📉", "Churn Classification", "Khách hàng nào sắp rời bỏ?",
            "Decision Tree, Random Forest, Naive Bayes",
            f"Dự đoán cho <b>{len(du_doan):,}</b> khách hàng; F1 tốt nhất "
            f"<b>{mo_hinh_tot_nhat['F1']:.4f}</b> ({mo_hinh_tot_nhat['Model name']})",
            "#ec4899"), unsafe_allow_html=True)

    # Hai biểu đồ kết quả nổi bật
    st.markdown("#### 🏆 Kết quả nổi bật")
    cot_trai, cot_phai = st.columns(2, gap="large")

    with cot_trai:
        bieu_do_tron = go.Figure(go.Pie(
            labels=doanh_thu.index.tolist(), values=doanh_thu.values.tolist(), hole=0.45,
            marker=dict(colors=[mau_phan_khuc(ten) for ten in doanh_thu.index]),
            textinfo="label+percent",
        ))
        bieu_do_tron.update_layout(title="Đóng góp giá trị chi tiêu theo phân khúc", showlegend=False)
        st.plotly_chart(dinh_dang_bieu_do(bieu_do_tron, 360), key="home_doanh_thu")

    with cot_phai:
        bieu_do_cot = go.Figure(go.Bar(
            x=ket_qua["Model name"], y=ket_qua["F1"],
            marker_color=[mau_mo_hinh(ten) for ten in ket_qua["Model name"]],
            text=[f"{gia_tri:.4f}" for gia_tri in ket_qua["F1"]], textposition="outside",
        ))
        bieu_do_cot.add_hline(y=f1_baseline, line_dash="dash", line_color="#475569",
                              annotation_text=f"Baseline F1 = {f1_baseline:.4f}",
                              annotation_position="bottom right")
        bieu_do_cot.update_yaxes(range=[0.5, min(1.0, float(ket_qua["F1"].max()) + 0.08)], title="F1 (trục bắt đầu từ 0,5)")
        bieu_do_cot.update_layout(title="F1 của các mô hình Churn so với baseline")
        st.plotly_chart(dinh_dang_bieu_do(bieu_do_cot, 360), key="home_f1")
    st.caption(
        "Baseline là mô hình ngây thơ luôn đoán 'khách sẽ rời bỏ'. "
        "Mức vượt baseline của các mô hình là khiêm tốn, xem chi tiết ở trang Đánh giá mô hình."
    )

    # Hướng dẫn sử dụng và thông tin dữ liệu
    st.divider()
    cot_a, cot_b = st.columns([1.2, 1], gap="large")
    with cot_a:
        st.markdown(
            """
            <div class="iq-card">
                <div class="section-title">🧭 Hướng dẫn sử dụng</div>
                <div class="muted">
                    <b>Phân khúc RFM</b>: Xem hồ sơ từng nhóm khách và cách các nhóm phân bố.<br>
                    <b>Dự đoán Churn</b>: Tra cứu khách trong tập Test, nhập khách mới
                    hoặc lấy danh sách khách cần chăm sóc ưu tiên.<br>
                    <b>Gợi ý Cross-sell</b>: Xem sản phẩm thường mua kèm theo phân khúc hoặc theo sản phẩm.<br>
                    <b>Đánh giá mô hình</b>: So sánh mô hình, đường ROC và thử đổi ngưỡng quyết định.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cot_b:
        st.markdown(
            f"""
            <div class="iq-card">
                <div class="section-title">📦 Dữ liệu và phương pháp</div>
                <div class="muted">
                    Dữ liệu thô <b>{THONG_TIN_DU_AN['so_dong_goc']:,}</b> dòng, còn
                    <b>{THONG_TIN_DU_AN['so_dong_sau_lam_sach']:,}</b> dòng sau làm sạch.<br>
                    Churn: không mua lại trong <b>{THONG_TIN_DU_AN['cua_so_churn_ngay']}</b> ngày;
                    Train/Test tách theo thời gian.
                </div>
                <div style="margin-top:.5rem;">
                    <span class="pill">CRISP-DM</span>
                    <span class="pill">SMOTE chỉ trên Train</span>
                    <span class="pill">GridSearchCV</span>
                    <span class="pill">Silhouette</span>
                    <span class="pill">ROC-AUC</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
