import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from Common import (
    ban_do_mau_phan_khuc, dinh_dang_bieu_do, hien_anh, hien_bang, hien_loi_thieu_tep,
    hien_tieu_de_trang, icon_phan_khuc, mau_phan_khuc, tai_bo_phan_khuc, tai_phan_khuc,
)


def _tong_hop_phan_khuc(df):
    bang = df.groupby("Tên phân khúc").agg(
        so_khach=("Customer ID", "nunique"),
        recency_tb=("Recency", "mean"),
        frequency_tb=("Frequency", "mean"),
        monetary_tb=("Monetary", "mean"),
        monetary_tong=("Monetary", "sum"),
    )
    bang["ty_le_khach"] = bang["so_khach"] / bang["so_khach"].sum() * 100
    bang["ty_trong_monetary"] = bang["monetary_tong"] / bang["monetary_tong"].sum() * 100
    return bang.sort_values("monetary_tb", ascending=False)


def _the_phan_khuc(ten: str, dong) -> str:
    mau = mau_phan_khuc(ten)
    return f"""
    <div class="sky-card fade-in" style="border-top:4px solid {mau};text-align:center;">
        <div style="font-size:2rem;">{icon_phan_khuc(ten)}</div>
        <div style="font-weight:800;font-size:1.05rem;color:{mau};">{ten}</div>
        <div style="font-size:1.7rem;font-weight:700;margin:.2rem 0;">{int(dong['so_khach']):,}</div>
        <div class="muted" style="font-size:.85rem;">{dong['ty_le_khach']:.1f}% số khách</div>
        <hr class="fancy-divider" style="margin:10px 0;">
        <div style="font-size:.88rem;line-height:1.7;text-align:left;">
            <b>Recency TB:</b> {dong['recency_tb']:.0f} ngày<br>
            <b>Frequency TB:</b> {dong['frequency_tb']:.1f} hoá đơn<br>
            <b>Monetary TB:</b> £{dong['monetary_tb']:,.0f}<br>
            <b>Đóng góp chi tiêu:</b> {dong['ty_trong_monetary']:.1f}%
        </div>
    </div>
    """


def _tab_tong_quan(df, tong_hop) -> None:
    ban_do_mau = ban_do_mau_phan_khuc(tong_hop.index)
    cot_trai, cot_phai = st.columns(2, gap="large")

    with cot_trai:
        bieu_do = px.bar(
            tong_hop.reset_index(), x="Tên phân khúc", y="so_khach", color="Tên phân khúc",
            color_discrete_map=ban_do_mau, text="so_khach", title="Số khách hàng từng phân khúc",
        )
        bieu_do.update_traces(textposition="outside")
        bieu_do.update_layout(showlegend=False, yaxis_title="Số khách hàng", xaxis_title="")
        st.plotly_chart(dinh_dang_bieu_do(bieu_do, 340), key="pk_so_khach")

    with cot_phai:
        bieu_do = go.Figure(go.Pie(
            labels=tong_hop.index.tolist(), values=tong_hop["monetary_tong"].tolist(), hole=0.45,
            marker=dict(colors=[ban_do_mau[ten] for ten in tong_hop.index]), textinfo="label+percent",
        ))
        bieu_do.update_layout(title="Đóng góp giá trị chi tiêu (Monetary)", showlegend=False)
        st.plotly_chart(dinh_dang_bieu_do(bieu_do, 340), key="pk_monetary")

    st.markdown("##### 📋 Hồ sơ phân khúc")
    bang = tong_hop.reset_index().rename(columns={
        "Tên phân khúc": "Phân khúc", "so_khach": "Số khách", "ty_le_khach": "% khách",
        "recency_tb": "Recency TB (ngày)", "frequency_tb": "Frequency TB",
        "monetary_tb": "Monetary TB (£)", "ty_trong_monetary": "% tổng Monetary",
    })
    hien_bang(bang[["Phân khúc", "Số khách", "% khách", "Recency TB (ngày)", "Frequency TB",
                    "Monetary TB (£)", "% tổng Monetary"]].round(1))

    # Kiểm tra bộ dựng lại K-Means có khớp với kết quả đã lưu ở Notebook 04 không
    bo_phan_khuc = tai_bo_phan_khuc()
    if bo_phan_khuc is None:
        st.warning("Không tìm thấy KMeans_RFM.pkl: phần nhập khách mới ở trang Dự đoán sẽ không gán được phân khúc.")
    elif bo_phan_khuc["do_khop"] >= 0.999:
        st.success(f"Kiểm tra đồng bộ: mô hình K-Means gán đúng phân khúc cho {bo_phan_khuc['do_khop'] * 100:.1f}% "
                   "khách hàng so với kết quả Notebook 04.")
    else:
        st.warning(f"Kiểm tra đồng bộ: chỉ {bo_phan_khuc['do_khop'] * 100:.1f}% khách hàng khớp với kết quả Notebook 04. "
                   "Hãy kiểm tra lại file KMeans_RFM.pkl và Customer_Segments_Result.csv có cùng một lần chạy không.")


def _tab_phan_bo(df) -> None:
    ban_do_mau = ban_do_mau_phan_khuc(df["Tên phân khúc"].unique())
    cot_chon, cot_log = st.columns([2, 1])
    with cot_chon:
        chi_so = st.selectbox("Chọn chỉ số", ["Recency", "Frequency", "Monetary"], index=2, key="pk_chi_so")
    with cot_log:
        st.write("")
        dung_log = st.checkbox("Dùng thang log (nên bật cho Frequency, Monetary)", value=True, key="pk_log")

    du_lieu = df[df[chi_so] > 0] if dung_log else df   # Thang log không biểu diễn được giá trị 0
    cot_h, cot_b = st.columns(2)
    with cot_h:
        bieu_do = px.histogram(du_lieu, x=chi_so, color="Tên phân khúc", nbins=50, opacity=0.8,
                               color_discrete_map=ban_do_mau, log_x=dung_log,
                               title=f"Histogram {chi_so}")
        bieu_do.update_layout(legend_title_text="Phân khúc", yaxis_title="Số khách hàng")
        st.plotly_chart(dinh_dang_bieu_do(bieu_do, 380), key="pk_hist")
    with cot_b:
        bieu_do = px.box(du_lieu, x="Tên phân khúc", y=chi_so, color="Tên phân khúc",
                         color_discrete_map=ban_do_mau, log_y=dung_log, points="outliers",
                         title=f"Boxplot {chi_so} theo phân khúc")
        bieu_do.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(dinh_dang_bieu_do(bieu_do, 380), key="pk_box")

    st.markdown("##### 📋 Thống kê mô tả")
    thong_ke = df.groupby("Tên phân khúc")[chi_so].describe().round(2).reset_index()
    hien_bang(thong_ke.rename(columns={"Tên phân khúc": "Phân khúc"}))


def _tab_khong_gian(df) -> None:
    ban_do_mau = ban_do_mau_phan_khuc(df["Tên phân khúc"].unique())
    so_mau = st.slider("Số khách hiển thị (lấy ngẫu nhiên)", 500, min(5000, len(df)), min(3000, len(df)),
                       step=500, key="pk_so_mau") if len(df) > 1000 else len(df)
    du_lieu = df[df["Monetary"] > 0].sample(min(so_mau, int((df["Monetary"] > 0).sum())), random_state=42)

    bieu_do = px.scatter(
        du_lieu, x="Frequency", y="Monetary", color="Tên phân khúc", color_discrete_map=ban_do_mau,
        hover_data=["Customer ID", "Recency"], log_x=True, log_y=True, opacity=0.6,
        title="Khách hàng theo Frequency và Monetary (thang log)",
    )
    bieu_do.update_layout(legend_title_text="Phân khúc")
    st.plotly_chart(dinh_dang_bieu_do(bieu_do, 520), key="pk_scatter")
    st.caption("Khách có Monetary = 0 không hiển thị được trên thang log. Di chuột vào điểm để xem Customer ID và Recency.")


def _tab_hinh_notebook() -> None:
    cac_hinh = [
        ("Elbow", "Chọn số cụm: Elbow và Silhouette"),
        ("KMeans vs KMedoids", "So sánh K-Means và K-Medoids (PAM)"),
        ("Segment Profiles", "Hồ sơ trung bình từng phân khúc"),
        ("RFM Distribution", "Phân phối RFM trước và sau log"),
    ]
    co_hinh = False
    for tu_khoa, chu_thich in cac_hinh:
        if hien_anh(tu_khoa, chu_thich):
            co_hinh = True
    if not co_hinh:
        st.info("Chưa tìm thấy ảnh trong thư mục Figures. Hãy chạy Notebook 04 để sinh biểu đồ.")


def Segments() -> None:
    hien_tieu_de_trang("💎", "Phân khúc khách hàng RFM",
                       "K-Means trên Recency, Frequency, Monetary (đã log và chuẩn hoá) - kết quả Notebook 04")
    try:
        df = tai_phan_khuc()
    except FileNotFoundError as loi:
        hien_loi_thieu_tep(loi)
        return

    tong_hop = _tong_hop_phan_khuc(df)

    # Thẻ hồ sơ từng phân khúc
    cot = st.columns(len(tong_hop), gap="medium")
    for cot_hien_tai, (ten, dong) in zip(cot, tong_hop.iterrows()):
        with cot_hien_tai:
            st.markdown(_the_phan_khuc(ten, dong), unsafe_allow_html=True)

    tab_tong_quan, tab_phan_bo, tab_khong_gian, tab_hinh = st.tabs([
        "Tổng quan", "Phân bố RFM", "Không gian khách hàng", "Hình từ notebook",
    ])
    with tab_tong_quan:
        _tab_tong_quan(df, tong_hop)
    with tab_phan_bo:
        _tab_phan_bo(df)
    with tab_khong_gian:
        _tab_khong_gian(df)
    with tab_hinh:
        _tab_hinh_notebook()
