import streamlit as st

THANH_VIEN = [
    {"ten": "Trần Văn Khang", "mssv": "52400199", "vai_tro": "Data & Infrastructure Lead",
     "nhiem_vu": "Notebook 01 và 02: Data Understanding, Data Preparation", "mau": "#0ea5e9"},
    
    {"ten": "Trần Tấn Lực", "mssv": "52400055", "vai_tro": "Pattern Mining Lead",
     "nhiem_vu": "Notebook 03 và 04: Market Basket Analysis, RFM Clustering", "mau": "#f59e0b"},
    
    {"ten": "Ngô Đức Huân", "mssv": "52400192", "vai_tro": "Classification & Integration Lead",
     "nhiem_vu": "Notebook 05 và 06: Churn Classification, Evaluation & Integration", "mau": "#22c55e"},
]

# Bảng các notebook của dự án
CAC_NOTEBOOK = [
    ("01", "Data Understanding", "Khám phá dữ liệu - Phát hiện 7 vấn đề chất lượng - Phân tích Outliers"),
    ("02", "Data Preparation", "Làm sạch 5 bước - Tạo RFM - Xuất dữ liệu đã xử lý"),
    ("03", "Market Basket Analysis", "Apriori so với FP-Growth - Luật kết hợp - Luật theo phân khúc"),
    ("04", "RFM Clustering", "K-Means so với K-Medoids (PAM) - Chọn số cụm - Đặt tên phân khúc"),
    ("05", "Churn Classification", "Nhãn Churn theo phân phối thực tế - SMOTE - 3 Thuật toán - GridSearchCV"),
    ("06", "Evaluation & Comparison", "Tổng hợp ba kỹ thuật và đối chiếu chéo phân khúc với Churn"),
]

CONG_NGHE = ["Python", "pandas", "NumPy", "scikit-learn", "imbalanced-learn", "mlxtend", "Plotly",
             "Streamlit", "joblib", "Apriori", "FP-Growth", "K-Means", "K-Medoids (PAM)", "Decision Tree",
             "Random Forest", "Naive Bayes", "SMOTE", "GridSearchCV"]


def AboutMe() -> None:
    st.markdown(
        """
        <div style="padding:28px;border-radius:20px;
                    background:linear-gradient(135deg,#38bdf8,#60a5fa,#93c5fd);
                    color:white;box-shadow:0 8px 25px rgba(56,189,248,.25);margin-bottom:16px;">
            <h1 style="margin-bottom:6px;color:white;">🛍️ Customer Analytics for E-Commerce</h1>
            <p style="font-size:18px;margin:0;">Đồ án cuối kỳ • Khai phá dữ liệu và Khai phá tri thức • Track B: Applied Project</p>
            <p style="margin:6px 0 0 0;">Market Basket • RFM Clustering • Churn Classification • Streamlit Demo</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Bốn thẻ thông tin nhanh
    the_nhanh = [
        ("📚", "Học phần", "Khai phá dữ liệu và Khai phá tri thức", "Đồ án cuối kỳ"),
        ("📊", "Dataset", "Online Retail II", "UCI · 2009-2011"),
        ("🧩", "Kỹ thuật", "3 góc nhìn", "MBA · RFM · Churn"),
        ("💻", "Demo", "Streamlit", "Python · Plotly"),
    ]
    for cot, (icon, tieu_de, gia_tri, phu) in zip(st.columns(4), the_nhanh):
        with cot:
            st.markdown(
                f"""
                <div class="sky-card fade-in" style="text-align:center;min-height:140px;">
                    <div style="font-size:32px;">{icon}</div>
                    <h4 style="margin:8px 0 4px 0;color:#0284c7;">{tieu_de}</h4>
                    <div style="font-size:19px;font-weight:700;color:#0f172a;">{gia_tri}</div>
                    <div style="font-size:13px;color:#64748b;margin-top:4px;">{phu}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.divider()

    # Các notebook và công nghệ
    cot_trai, cot_phai = st.columns([1.5, 1], gap="large")
    with cot_trai:
        dong_notebook = "".join(
            f'<div style="margin-bottom:.5rem;"><b class="highlight-text">Notebook {so}: {ten}</b><br>'
            f'<span class="muted" style="font-size:.9rem;">{mo_ta}</span></div>'
            for so, ten, mo_ta in CAC_NOTEBOOK
        )
        st.markdown(
            f"""
            <div class="iq-card fade-in">
                <div class="section-title">📓 Các Notebooks của dự án</div>
                {dong_notebook}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cot_phai:
        nhan_cong_nghe = "".join(f'<span class="pill">{ten}</span>' for ten in CONG_NGHE)
        st.markdown(
            f"""
            <div class="iq-card fade-in">
                <div class="section-title">🛠️ Công nghệ sử dụng</div>
                <div>{nhan_cong_nghe}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="iq-card fade-in">
                <div class="section-title">🔗 Nguồn dữ liệu</div>
                <div class="muted">
                    <a href="https://archive.ics.uci.edu/dataset/502/online+retail+ii" target="_blank">
                        UCI Machine Learning Repository - Online Retail II
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()

    # Thành viên nhóm
    st.markdown(
        '<div style="font-size:1.25rem;font-weight:700;color:#0369a1;margin-bottom:1rem;">👥 Thành viên nhóm</div>',
        unsafe_allow_html=True,
    )
    for cot, thanh_vien in zip(st.columns(len(THANH_VIEN), gap="large"), THANH_VIEN):
        with cot:
            st.markdown(
                f"""
                <div class="sky-card fade-in" style="text-align:center;border-top:4px solid {thanh_vien['mau']};">
                    <div style="font-weight:800;font-size:1.08rem;color:{thanh_vien['mau']};">{thanh_vien['ten']}</div>
                    <div style="opacity:.65;font-size:.88rem;margin:.25rem 0;">MSSV: <b>{thanh_vien['mssv']}</b></div>
                    <div style="margin:.4rem 0;">
                        <span class="pill" style="border-color:{thanh_vien['mau']};color:{thanh_vien['mau']};">
                            {thanh_vien['vai_tro']}
                        </span>
                    </div>
                    <hr style="margin:.6rem 0;opacity:.18;">
                    <div class="section-title" style="font-size:.88rem;">Nhiệm vụ</div>
                    <div class="muted" style="font-size:.85rem;">{thanh_vien['nhiem_vu']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
