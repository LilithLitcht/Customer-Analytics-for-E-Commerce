# -*- coding: utf-8 -*-
"""
Main.py
Điểm vào của ứng dụng Demo "Customer Analytics for E-Commerce".
Chạy từ thư mục gốc dự án:   streamlit run Demo/Main.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Đảm bảo import được các file cùng thư mục Demo dù chạy từ đâu
THU_MUC_DEMO = Path(__file__).resolve().parent
if str(THU_MUC_DEMO) not in sys.path:
    sys.path.insert(0, str(THU_MUC_DEMO))

import streamlit as st
from streamlit_option_menu import option_menu

# Cấu hình trang: bắt buộc là lệnh Streamlit đầu tiên
st.set_page_config(
    page_title="Customer Analytics - Demo",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from AboutMe import AboutMe
from Evaluation import Evaluation
from Home import Home
from Predict import Predict
from Recommend import Recommend
from Segments import Segments

# Danh sách trang: (tên hiển thị, biểu tượng Bootstrap, hàm vẽ trang)
CAC_TRANG = [
    ("Tổng quan", "house-fill", Home),
    ("Phân khúc RFM", "diagram-3-fill", Segments),
    ("Dự đoán Churn", "graph-up-arrow", Predict),
    ("Gợi ý Cross-sell", "cart-check-fill", Recommend),
    ("Đánh giá mô hình", "bar-chart-fill", Evaluation),
    ("Về nhóm", "people-fill", AboutMe),
]


def _nap_css() -> None:
    """Nạp file Styles.css (nếu có) để áp dụng giao diện."""
    duong_dan = THU_MUC_DEMO / "Styles.css"
    if duong_dan.exists():
        with open(duong_dan, encoding="utf-8") as tep:
            st.markdown(f"<style>{tep.read()}</style>", unsafe_allow_html=True)


def _ve_thanh_ben() -> str:
    """Vẽ thanh bên gồm logo, menu điều hướng, chân trang; trả về tên trang đang chọn."""
    with st.sidebar:
        # Logo: dùng Logo.png trong thư mục Demo nếu có, không thì hiển thị tiêu đề chữ
        duong_dan_logo = THU_MUC_DEMO / "Logo.png"
        if duong_dan_logo.exists():
            st.image(str(duong_dan_logo))
        else:
            st.markdown(
                """
                <div style="text-align:center; padding:16px 0 8px 0;">
                    <div style="font-size:2.8rem;">🛍️</div>
                    <div style="font-weight:700; font-size:1.1rem; color:#fff; letter-spacing:.03em;">
                        Customer Analytics
                    </div>
                    <div style="font-size:.78rem; color:#bae6fd; margin-top:.2rem;">
                        Online Retail II · Streamlit
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Menu điều hướng
        trang_chon = option_menu(
            menu_title="Menu chính",
            options=[ten for ten, _, _ in CAC_TRANG],
            icons=[icon for _, icon, _ in CAC_TRANG],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "8px",
                    "background-color": "#fdf2f8",
                    "border-radius": "14px",
                    "border": "1px solid #f9a8d4",
                },
                "icon": {"color": "#7c3aed", "font-size": "16px"},
                "menu-title": {"color": "#7c3aed", "font-weight": "700"},
                "nav-link": {
                    "font-size": "15px",
                    "color": "#5b21b6",
                    "background-color": "#fdf2f8",
                    "margin": "4px 0",
                    "border-radius": "10px",
                    "padding": "10px 14px",
                },
                "nav-link-hover": {"background-color": "#fbcfe8", "color": "#4c1d95"},
                "nav-link-selected": {
                    "background-color": "#ec4899",
                    "color": "white",
                    "font-weight": "700",
                    "border-radius": "10px",
                },
            },
        )

        # Chân thanh bên
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:.70rem; color:#7dd3fc; text-align:center; opacity:.80; line-height:1.6;">
                Khai phá dữ liệu (IT701220)<br>
                Đồ án cuối kỳ · Track B<br>
                scikit-learn · Plotly · Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )
    return trang_chon


def main() -> None:
    """Nạp giao diện, vẽ thanh bên rồi hiển thị đúng trang được chọn."""
    _nap_css()
    trang_chon = _ve_thanh_ben()

    # Tìm hàm vẽ trang tương ứng; mặc định là trang Tổng quan
    cac_ham = {ten: ham for ten, _, ham in CAC_TRANG}
    cac_ham.get(trang_chon, Home)()


if __name__ == "__main__":
    main()
