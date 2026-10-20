import pandas as pd
import streamlit as st

from Common import (
    DAC_TRUNG, KICH_BAN_MAU, MO_HINH_MAC_DINH, MO_TA_DAC_TRUNG, THONG_TIN_DU_AN,
    dinh_dang_bang_luat, hien_bang, hien_loi_thieu_tep, hien_tieu_de_trang, lay_goi_y_phan_khuc,
    lay_hanh_dong, phan_loai_rui_ro, tai_du_doan, tai_mo_hinh, ve_dong_ho_rui_ro,
    xac_dinh_phan_khuc,
)

# Các ô nhập liệu của tab "Nhập khách hàng mới" (AvgOrderValue được tự tính nên không có ô riêng)
KHOA_NHAP = ["Recency", "Frequency", "Monetary", "Tenure", "AvgGapDays"]
KICH_BAN_BAN_DAU = "Khách phổ thông (Mua thỉnh thoảng)"


def hien_ket_qua_du_doan(dac_trung: dict, ten_phan_khuc: str, ten_mo_hinh: str, mo_hinh: dict,
                         khoa: str, nhan_thuc_te: int = None, xac_suat_da_luu: float = None) -> None:
    dau_vao = pd.DataFrame([dac_trung])[DAC_TRUNG].astype(float)

    ket_qua = {}
    for ten, mo_hinh_con in mo_hinh.items():
        ket_qua[ten] = (float(mo_hinh_con.predict_proba(dau_vao)[0, 1]), int(mo_hinh_con.predict(dau_vao)[0]))

    xac_suat, nhan_du_doan = ket_qua[ten_mo_hinh]
    rui_ro = phan_loai_rui_ro(xac_suat)
    cua_so = THONG_TIN_DU_AN["cua_so_churn_ngay"]
    cau_du_doan = (f"Dự đoán: khách sẽ KHÔNG quay lại trong {cua_so} ngày tới" if nhan_du_doan == 1
                   else f"Dự đoán: khách sẽ quay lại mua trong {cua_so} ngày tới")

    dong_thuc_te = ""
    if nhan_thuc_te is not None:
        dung = int(nhan_thuc_te) == nhan_du_doan
        thuc_te = "Đã rời bỏ" if int(nhan_thuc_te) == 1 else "Đã quay lại mua"
        mau_ket_luan = "#16a34a" if dung else "#dc2626"
        dong_thuc_te = (f'<div style="margin-top:10px;font-weight:700;color:{mau_ket_luan};">'
                        f'Thực tế (tập Test): {thuc_te} - dự đoán {"ĐÚNG" if dung else "SAI"}</div>')

    cot_the, cot_dong_ho = st.columns([1, 1.1])
    with cot_the:
        st.markdown(
            f"""
            <div class="result-card" style="background:{rui_ro['nen']};border:2px solid {rui_ro['mau']};">
                <div style="font-size:3rem;">{rui_ro['icon']}</div>
                <div style="font-size:1.6rem;font-weight:800;color:{rui_ro['mau']};margin:.4rem 0;">
                    Rủi ro rời bỏ: {rui_ro['nhan']}
                </div>
                <div style="color:#475569;font-size:.95rem;">{rui_ro['mo_ta']}</div>
                <div style="margin-top:10px;font-weight:600;color:#0f172a;">{cau_du_doan}</div>
                <div style="margin-top:10px;">
                    <span class="pill">Mô hình: {ten_mo_hinh}</span>
                    <span class="pill">Phân khúc: {ten_phan_khuc}</span>
                </div>
                {dong_thuc_te}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cot_dong_ho:
        st.plotly_chart(ve_dong_ho_rui_ro(xac_suat), key=f"{khoa}_dong_ho")

    st.caption(
        "Lưu ý: xác suất của Naive Bayes thường không được hiệu chuẩn (hay sát 0 hoặc 1), "
        "nên hãy đọc như điểm xếp hạng rủi ro hơn là xác suất thực."
    )
    if xac_suat_da_luu is not None:
        st.caption(f"Đối chiếu: xác suất đã lưu trong file kết quả = {xac_suat_da_luu:.4f}; "
                   f"tính lại từ file mô hình .pkl = {xac_suat:.4f}.")

    if len(ket_qua) > 1:
        st.markdown("##### So sánh các mô hình")
        bang = pd.DataFrame([
            {"Mô hình": ten, "Xác suất Churn": f"{p * 100:.1f}%",
             "Dự đoán": "Rời bỏ" if nhan == 1 else "Quay lại"}
            for ten, (p, nhan) in ket_qua.items()
        ])
        hien_bang(bang)
        if len({nhan for _, nhan in ket_qua.values()}) == 1:
            st.success(f"Cả {len(ket_qua)} mô hình đồng thuận: {'Rời bỏ' if nhan_du_doan == 1 else 'Quay lại'}.")
        else:
            st.warning("Các mô hình cho kết quả khác nhau, kết luận cho khách này chưa chắc chắn.")

    # Hành động đề xuất theo phân khúc và mức rủi ro
    st.markdown(
        f"""
        <div class="iq-card">
            <div class="section-title">🎯 Hành động đề xuất</div>
            <div class="muted">{lay_hanh_dong(ten_phan_khuc, xac_suat)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    goi_y = lay_goi_y_phan_khuc(ten_phan_khuc, so_luat=3)
    if not goi_y.empty:
        st.markdown(f"##### 🛒 Gợi ý bán kèm cho phân khúc {ten_phan_khuc}")
        hien_bang(dinh_dang_bang_luat(goi_y))
        st.caption("Ba luật 1 sản phẩm → 1 sản phẩm có Lift cao nhất của phân khúc. "
                   "Đây là gợi ý theo nhóm, chưa phải cá nhân hoá từng khách.")

    with st.expander("Xem lại thông số của khách hàng"):
        bang = pd.DataFrame({
            "Đặc trưng": DAC_TRUNG,
            "Giá trị": [round(float(dac_trung[c]), 2) for c in DAC_TRUNG],
            "Ý nghĩa": [MO_TA_DAC_TRUNG[c] for c in DAC_TRUNG],
        })
        hien_bang(bang)


def _tab_tra_cuu(du_doan: pd.DataFrame, mo_hinh: dict) -> None:
    st.markdown(
        """
        <div class="sky-card fade-in">
            <div style="font-size:1.1rem;font-weight:700;color:#0369a1;">Tra cứu khách hàng trong tập Test</div>
            <div class="muted" style="margin-top:.3rem;">
                Chọn một khách hàng để xem mô hình dự đoán thế nào và đối chiếu với kết quả thực tế.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ten_mo_hinh_ds = list(mo_hinh.keys())
    cot_ma, cot_mo_hinh = st.columns([1, 1])
    with cot_ma:
        ma_khach = st.selectbox("Mã khách hàng (Customer ID)", sorted(du_doan["Customer ID"].tolist()), key="tc_ma_khach")
    with cot_mo_hinh:
        chi_so = ten_mo_hinh_ds.index(MO_HINH_MAC_DINH) if MO_HINH_MAC_DINH in ten_mo_hinh_ds else 0
        ten_mo_hinh = st.selectbox("Mô hình dự đoán", ten_mo_hinh_ds, index=chi_so, key="tc_mo_hinh")

    dong = du_doan[du_doan["Customer ID"] == ma_khach].iloc[0]
    dac_trung = {c: float(dong[c]) for c in DAC_TRUNG}

    # Thông tin RFM và các đặc trưng của khách
    cot = st.columns(6)
    cot[0].metric("Recency", f"{dac_trung['Recency']:.0f} ngày")
    cot[1].metric("Frequency", f"{dac_trung['Frequency']:.0f}")
    cot[2].metric("Monetary", f"£{dac_trung['Monetary']:,.0f}")
    cot[3].metric("Avg Order Value", f"£{dac_trung['AvgOrderValue']:,.0f}")
    cot[4].metric("Tenure", f"{dac_trung['Tenure']:.0f} ngày")
    cot[5].metric("Avg Gap", f"{dac_trung['AvgGapDays']:.0f} ngày")

    # Chỉ đối chiếu xác suất đã lưu khi đang chọn đúng mô hình đã dùng để xuất file
    xac_suat_da_luu = float(dong["Xác suất Churn"]) if ten_mo_hinh == dong.get("Mô hình sử dụng") else None
    hien_ket_qua_du_doan(dac_trung, str(dong["Tên phân khúc"]), ten_mo_hinh, mo_hinh, "tra_cuu",
                         nhan_thuc_te=int(dong["Churn thực tế"]), xac_suat_da_luu=xac_suat_da_luu)


def _khoi_tao_gia_tri_nhap() -> None:
    for khoa, gia_tri in KICH_BAN_MAU[KICH_BAN_BAN_DAU].items():
        st.session_state.setdefault(f"nhap_{khoa}", gia_tri)


def _ap_dung_kich_ban() -> None:
    ten = st.session_state.get("nhap_kich_ban")
    if ten in KICH_BAN_MAU:
        for khoa, gia_tri in KICH_BAN_MAU[ten].items():
            st.session_state[f"nhap_{khoa}"] = gia_tri


def _tab_nhap_moi(mo_hinh: dict) -> None:
    st.markdown(
        """
        <div class="sky-card fade-in">
            <div style="font-size:1.1rem;font-weight:700;color:#0369a1;">Nhập thông số khách hàng mới</div>
            <div class="muted" style="margin-top:.3rem;">
                Chọn kịch bản mẫu hoặc tự nhập → hệ thống gán phân khúc, ước lượng rủi ro rời bỏ và gợi ý bán kèm.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _khoi_tao_gia_tri_nhap()

    cot_kich_ban, cot_mo_hinh = st.columns([1.3, 1])
    with cot_kich_ban:
        st.selectbox("Kịch bản mẫu", ["Tự nhập"] + list(KICH_BAN_MAU), key="nhap_kich_ban",
                     on_change=_ap_dung_kich_ban, help="Chọn để điền sẵn các ô bên dưới, sau đó vẫn sửa được.")
    with cot_mo_hinh:
        ten_mo_hinh_ds = list(mo_hinh.keys())
        chi_so = ten_mo_hinh_ds.index(MO_HINH_MAC_DINH) if MO_HINH_MAC_DINH in ten_mo_hinh_ds else 0
        ten_mo_hinh = st.selectbox("Mô hình dự đoán", ten_mo_hinh_ds, index=chi_so, key="nhap_mo_hinh")

    st.markdown("##### Thông số khách hàng")
    cot1, cot2, cot3 = st.columns(3)
    with cot1:
        recency = st.number_input("Recency (ngày)", min_value=0, max_value=800, step=1, key="nhap_Recency",
                                  help=MO_TA_DAC_TRUNG["Recency"])
        tenure = st.number_input("Tenure (ngày)", min_value=0, max_value=800, step=1, key="nhap_Tenure",
                                 help=MO_TA_DAC_TRUNG["Tenure"])
    with cot2:
        frequency = st.number_input("Frequency (số hoá đơn)", min_value=1, max_value=500, step=1,
                                    key="nhap_Frequency", help=MO_TA_DAC_TRUNG["Frequency"])
        # Lần mua đầu không thể sau lần mua gần nhất nên Tenure phải >= Recency
        tenure_hop_le = max(int(tenure), int(recency))
        # Khách chỉ mua 1 lần: khi huấn luyện AvgGapDays được điền bằng Tenure, ở đây làm giống vậy
        if frequency == 1:
            st.session_state["nhap_AvgGapDays"] = float(tenure_hop_le)
        khoang_cach = st.number_input("Avg Gap (ngày giữa các lần mua)", min_value=0.0, max_value=800.0,
                                      step=1.0, format="%.1f", key="nhap_AvgGapDays",
                                      disabled=(frequency == 1), help=MO_TA_DAC_TRUNG["AvgGapDays"])
    with cot3:
        monetary = st.number_input("Monetary (£)", min_value=0.0, max_value=1_000_000.0, step=100.0,
                                   format="%.2f", key="nhap_Monetary", help=MO_TA_DAC_TRUNG["Monetary"])
        st.metric("Avg Order Value (tự tính)", f"£{monetary / frequency:,.2f}")

    if tenure < recency:
        st.warning("Tenure không thể nhỏ hơn Recency (lần mua đầu không thể sau lần mua gần nhất). "
                   "Hệ thống tạm dùng Tenure = Recency.")
    if frequency == 1:
        st.info("Khách chỉ mua 1 lần nên Avg Gap được tự điền bằng Tenure, đúng như cách huấn luyện mô hình.")

    dac_trung = {
        "Recency": float(recency), "Frequency": float(frequency), "Monetary": float(monetary),
        "AvgOrderValue": float(monetary) / float(frequency),
        "Tenure": float(tenure_hop_le), "AvgGapDays": float(khoang_cach),
    }

    # Gán phân khúc bằng K-Means của Notebook 04
    ten_phan_khuc = xac_dinh_phan_khuc(recency, frequency, monetary)
    if ten_phan_khuc is None:
        st.info("Không tìm thấy KMeans_RFM.pkl nên chưa gán được phân khúc; vẫn dự đoán Churn bình thường.")
        ten_phan_khuc = "Không xác định"

    st.markdown("---")
    st.markdown("#### Kết quả")
    hien_ket_qua_du_doan(dac_trung, ten_phan_khuc, ten_mo_hinh, mo_hinh, "khach_moi")


def _tab_danh_sach(du_doan: pd.DataFrame) -> None:
    ten_mo_hinh_luu = str(du_doan["Mô hình sử dụng"].iloc[0]) if "Mô hình sử dụng" in du_doan.columns else "mô hình đã chọn"
    st.markdown(
        f"""
        <div class="sky-card fade-in">
            <div style="font-size:1.1rem;font-weight:700;color:#0369a1;">Danh sách khách cần chăm sóc ưu tiên</div>
            <div class="muted" style="margin-top:.3rem;">
                Lọc khách có xác suất Churn cao từ file dự đoán ({ten_mo_hinh_luu}, tập Test).
                Nên ưu tiên khách có giá trị chi tiêu lớn để chiến dịch giữ chân hiệu quả nhất.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ds_phan_khuc = sorted(du_doan["Tên phân khúc"].unique())
    cot1, cot2 = st.columns(2)
    with cot1:
        chon_phan_khuc = st.multiselect("Phân khúc", ds_phan_khuc, default=ds_phan_khuc, key="ds_phan_khuc")
        cach_xep = st.radio("Ưu tiên theo", ["Giá trị chi tiêu (Monetary) cao nhất", "Xác suất Churn cao nhất"],
                            horizontal=True, key="ds_cach_xep")
    with cot2:
        nguong = st.slider("Xác suất Churn tối thiểu", 0.0, 1.0, 0.7, 0.05, key="ds_nguong")
        so_dong = st.slider("Số khách hiển thị", 10, 200, 50, 10, key="ds_so_dong")

    loc = du_doan[du_doan["Tên phân khúc"].isin(chon_phan_khuc) & (du_doan["Xác suất Churn"] >= nguong)]
    if loc.empty:
        st.warning("Không có khách hàng nào thoả điều kiện lọc. Hãy giảm ngưỡng xác suất hoặc chọn thêm phân khúc.")
        return

    k1, k2, k3 = st.columns(3)
    k1.metric("Số khách thoả điều kiện", f"{len(loc):,}")
    k2.metric("Tổng chi tiêu của nhóm này", f"£{loc['Monetary'].sum():,.0f}")
    k3.metric("Tỷ lệ thực tế đã rời bỏ", f"{loc['Churn thực tế'].mean() * 100:.1f}%",
              help="Đối chiếu với nhãn thật trong tập Test: càng cao nghĩa là danh sách càng chính xác")

    cot_xep = "Monetary" if cach_xep.startswith("Giá trị") else "Xác suất Churn"
    danh_sach = loc.sort_values(cot_xep, ascending=False).head(so_dong)

    bang = pd.DataFrame({
        "Customer ID": danh_sach["Customer ID"],
        "Phân khúc": danh_sach["Tên phân khúc"],
        "Recency (ngày)": danh_sach["Recency"],
        "Frequency": danh_sach["Frequency"],
        "Monetary (£)": danh_sach["Monetary"].round(2),
        "Xác suất Churn": danh_sach["Xác suất Churn"],
        "Thực tế (Test)": danh_sach["Churn thực tế"].map({1: "Đã rời bỏ", 0: "Đã quay lại"}),
    })
    hien_bang(bang)

    st.download_button(
        "Tải danh sách (CSV)", data=bang.to_csv(index=False).encode("utf-8-sig"),
        file_name="Danh_sach_khach_can_cham_soc.csv", mime="text/csv", type="primary", key="ds_tai_ve",
    )


def Predict() -> None:
    cua_so = THONG_TIN_DU_AN["cua_so_churn_ngay"]
    hien_tieu_de_trang("📉", "Dự đoán khách hàng rời bỏ (Churn)",
                       f"Khách không mua lại trong {cua_so} ngày được coi là rời bỏ - kết quả Notebook 05")
    try:
        du_doan = tai_du_doan()
    except FileNotFoundError as loi:
        hien_loi_thieu_tep(loi)
        return

    mo_hinh = tai_mo_hinh()
    if not mo_hinh:
        st.error("Không tìm thấy file mô hình Churn (.pkl) trong thư mục Models.")
        st.info("Hãy chạy Notebook 05 để huấn luyện và lưu mô hình, rồi tải lại trang.")
        return

    tab_tra_cuu, tab_nhap_moi, tab_danh_sach = st.tabs([
        "Tra cứu khách hàng", "Nhập khách hàng mới", "Danh sách ưu tiên chăm sóc",
    ])
    with tab_tra_cuu:
        _tab_tra_cuu(du_doan, mo_hinh)
    with tab_nhap_moi:
        _tab_nhap_moi(mo_hinh)
    with tab_danh_sach:
        _tab_danh_sach(du_doan)
