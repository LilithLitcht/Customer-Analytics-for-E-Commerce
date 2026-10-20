import plotly.express as px
import streamlit as st

from Common import (
    dinh_dang_bang_luat, dinh_dang_bieu_do, hien_bang, hien_loi_thieu_tep, hien_tieu_de_trang,
    loc_luat, mau_phan_khuc, tai_luat_phan_khuc, tai_luat_tong,
)


def _ve_cot_ngang(df_luat, cot_gia_tri: str, mau: str, tieu_de: str, khoa: str) -> None:
    du_lieu = df_luat.copy()
    du_lieu["Luật"] = du_lieu["Antecedents Name"].str.slice(0, 40) + " → " + du_lieu["Consequents Name"].str.slice(0, 40)
    du_lieu = du_lieu.iloc[::-1]   
    bieu_do = px.bar(du_lieu, x=cot_gia_tri, y="Luật", orientation="h", text=cot_gia_tri, title=tieu_de)
    bieu_do.update_traces(marker_color=mau, texttemplate="%{text:.2f}", textposition="outside")
    bieu_do.update_layout(yaxis_title="", xaxis_title=cot_gia_tri)
    st.plotly_chart(dinh_dang_bieu_do(bieu_do, max(300, 60 + 34 * len(du_lieu))), key=khoa)


def _tab_theo_phan_khuc(luat_phan_khuc) -> None:
    ds_phan_khuc = sorted(luat_phan_khuc["Tên phân khúc"].unique())
    cot1, cot2 = st.columns([1.3, 1])
    with cot1:
        phan_khuc = st.selectbox("Phân khúc khách hàng", ds_phan_khuc, key="rc_phan_khuc")
        chi_1_1 = st.checkbox("Chỉ luật dạng 1 sản phẩm → 1 sản phẩm (dễ dùng để gợi ý)", value=True, key="rc_chi_1_1")
    with cot2:
        so_luat = st.slider("Số luật hiển thị", 5, 20, 10, key="rc_so_luat")
        lift_toi_thieu = st.slider("Lift tối thiểu", 1.0, 50.0, 3.0, 0.5, key="rc_lift")
        kulczynski_toi_thieu = st.slider("Kulczynski tối thiểu", 0.0, 1.0, 0.5, 0.05, key="rc_kulc")

    luat = luat_phan_khuc[luat_phan_khuc["Tên phân khúc"] == phan_khuc]
    thoa = loc_luat(luat, chi_1_1=chi_1_1, lift_toi_thieu=lift_toi_thieu, kulczynski_toi_thieu=kulczynski_toi_thieu)

    st.markdown(f"**{len(thoa):,}** luật thoả điều kiện trong tổng số **{len(luat):,}** luật của phân khúc này.")
    if thoa.empty:
        st.warning("Không có luật nào thoả điều kiện. Hãy giảm Lift hoặc Kulczynski tối thiểu.")
        return

    hien_thi = thoa.head(so_luat)
    hien_bang(dinh_dang_bang_luat(hien_thi))
    _ve_cot_ngang(hien_thi, "Lift", mau_phan_khuc(phan_khuc), f"Lift của các luật mạnh nhất - {phan_khuc}", "rc_bieu_do_pk")
    st.caption(
        "Lift cho biết mua sản phẩm này làm tăng khả năng mua sản phẩm kia gấp bao nhiêu lần so với ngẫu nhiên; "
        "Kulczynski cân bằng hai chiều của luật nên ít bị ảnh hưởng bởi sản phẩm quá phổ biến. "
        "Các luật mạnh nhất thường là mua theo bộ sưu tập (cùng dòng sản phẩm), phù hợp để bán theo bộ, "
        "chưa chứng minh quan hệ nhân quả."
    )


def _tab_theo_san_pham(luat_tong) -> None:
    # Danh sách sản phẩm: các sản phẩm đứng một mình ở vế trái của luật
    luat_don = luat_tong[luat_tong["So SP ve trai"] == 1]
    ds_san_pham = sorted(luat_don["Antecedents Name"].unique())
    if not ds_san_pham:
        st.info("Không có luật nào có vế trái là một sản phẩm.")
        return

    cot1, cot2 = st.columns([1.6, 1])
    with cot1:
        san_pham = st.selectbox("Chọn sản phẩm (gõ để tìm)", ds_san_pham, key="rc_san_pham")
    with cot2:
        confidence_toi_thieu = st.slider("Confidence tối thiểu", 0.0, 1.0, 0.3, 0.05, key="rc_conf")
        lift_toi_thieu = st.slider("Lift tối thiểu", 1.0, 50.0, 2.0, 0.5, key="rc_lift_sp")

    luat = luat_don[luat_don["Antecedents Name"] == san_pham]
    thoa = loc_luat(luat, chi_1_1=False, lift_toi_thieu=lift_toi_thieu,
                    confidence_toi_thieu=confidence_toi_thieu, bo_trung_chieu=False)
    st.markdown(f"Khách mua **{san_pham}** thường mua kèm: **{len(thoa):,}** luật thoả điều kiện.")
    if thoa.empty:
        st.warning("Không có luật nào thoả điều kiện. Hãy giảm Confidence hoặc Lift tối thiểu.")
        return

    hien_thi = thoa.head(10)
    hien_bang(dinh_dang_bang_luat(hien_thi))
    _ve_cot_ngang(hien_thi, "Confidence", "#0ea5e9", "Xác suất mua kèm (Confidence)", "rc_bieu_do_sp")
    st.caption("Confidence = trong các hoá đơn có sản phẩm bên trái, tỷ lệ hoá đơn cũng có sản phẩm bên phải. "
               "Luật này tính trên toàn bộ hoá đơn, kể cả hoá đơn không có Customer ID.")


def _tab_tong_quan_luat(luat_tong) -> None:
    luat = loc_luat(luat_tong, chi_1_1=True)
    so_mau = min(3000, len(luat))
    if so_mau == 0:
        st.info("Không có luật 1 → 1 để hiển thị.")
        return
    du_lieu = luat.sample(so_mau, random_state=42)

    bieu_do = px.scatter(
        du_lieu, x="Support", y="Confidence", color="Lift", log_x=True, color_continuous_scale="Blues",
        hover_data=["Antecedents Name", "Consequents Name", "Kulczynski"], opacity=0.7,
        title=f"Luật kết hợp 1 → 1 (mẫu {so_mau:,} luật)",
    )
    st.plotly_chart(dinh_dang_bieu_do(bieu_do, 480), key="rc_tong_quan")
    st.caption("Luật có Lift rất cao thường nằm ở vùng Support thấp (sản phẩm hiếm đi cùng nhau); "
               "luật ở vùng Support cao và Confidence cao có độ tin cậy thực tế tốt hơn nhưng Lift thường thấp.")


def Recommend() -> None:
    hien_tieu_de_trang("🛒", "Gợi ý bán kèm (Cross-sell)",
                       "Luật kết hợp từ FP-Growth: khách mua sản phẩm này thường mua kèm sản phẩm nào - Notebook 03")
    try:
        luat_phan_khuc = tai_luat_phan_khuc()
        luat_tong = tai_luat_tong()
    except FileNotFoundError as loi:
        hien_loi_thieu_tep(loi)
        return

    tab_phan_khuc, tab_san_pham, tab_tong_quan = st.tabs(["Theo phân khúc", "Theo sản phẩm", "Tổng quan luật"])
    with tab_phan_khuc:
        _tab_theo_phan_khuc(luat_phan_khuc)
    with tab_san_pham:
        _tab_theo_san_pham(luat_tong)
    with tab_tong_quan:
        _tab_tong_quan_luat(luat_tong)
