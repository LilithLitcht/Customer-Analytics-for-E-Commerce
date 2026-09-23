# Chứa các hàm vẽ biểu đồ 

from pathlib import Path
from .IO import FIGURES_DIR

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns


# Thiết lập Style dùng chung cho toàn bộ biểu đồ
PRIMARY_COLOR   = "#2E86AB"      # Biểu đồ đơn sắc (Histogram, Bar Chart 1 nhóm)
SECONDARY_COLOR = "#E67E22"      # Biểu đồ 2 màu (Trước/sau xử lý)
ALERT_COLOR     = "#C0392B"      # Biểu đồ cảnh báo (Outlier, Missing Values)

# Kích thước biểu đồ mặc định
SINGLE_PLOT_FIGSIZE = (8, 5)   
DOUBLE_PLOT_FIGSIZE = (14, 5)    
FIGURE_SAVE_DPI = 150 

# Hàm thiết lập style dùng chung cho toàn bộ biểu đồ
def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 100     
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"


# Hàm lưu biểu đồ
def save_current_figure(filename: str, output_dir: str | Path = None) -> Path:

    if output_dir is None:
        output_dir = FIGURES_DIR
    else:
        output_dir = Path(output_dir)

    # Tự động tạo thư mục Figures/ nếu chưa tồn tại
    output_dir.mkdir(parents=True, exist_ok=True)

    full_output_path = output_dir / filename

    # Cắt bớt lề thừa quanh biểu đồ, tránh ảnh bị hụt chữ
    plt.savefig(full_output_path, dpi=FIGURE_SAVE_DPI, bbox_inches="tight")
    print(f"Đã lưu biểu đồ: {full_output_path}")

    return full_output_path


# Hàm vẽ Histogram + Boxplot cho 1 cột số
def plot_histogram_boxplot(df: pd.DataFrame,
                            column: str,
                            title: str = None,
                            xlim: tuple = None,
                            save_as: str = None) -> None:

    if title is None:
        title = f"Phân phối của {column}"

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=DOUBLE_PLOT_FIGSIZE)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Biểu đồ 1: Histogram
    sns.histplot(data=df, x=column, bins=50, color=PRIMARY_COLOR, ax=ax_hist)
    ax_hist.set_title("Histogram (Hình dạng phân phối)")
    ax_hist.set_xlabel(column)
    ax_hist.set_ylabel("Số lượng dòng (Tần suất)")
    if xlim is not None:
        ax_hist.set_xlim(xlim)

    # Biểu đồ 2: Boxplot 
    sns.boxplot(data=df, x=column, color=SECONDARY_COLOR, ax=ax_box)
    ax_box.set_title("Boxplot (Phát hiện Outlier trực quan)")
    ax_box.set_xlabel(column)
    if xlim is not None:
        ax_box.set_xlim(xlim)

    plt.tight_layout()

    # Lưu ảnh trước khi show
    if save_as is not None:
        save_current_figure(save_as)
        
    plt.show()


# Hàm vẽ % Missing Values theo từng cột
def plot_missing_values(df: pd.DataFrame,
                         title: str = "Tỷ lệ giá trị thiếu (Missing Values) theo cột",
                         save_as: str = None) -> pd.DataFrame:

    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    missing_summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct
    }).sort_values("missing_pct", ascending=False)

    # Chỉ vẽ biểu đồ cho các cột có thiếu dữ liệu (> 0%)
    columns_with_missing = missing_summary[missing_summary["missing_pct"] > 0]

    if columns_with_missing.empty:
        print("Không có cột nào bị thiếu dữ liệu")
        return missing_summary

    fig, ax = plt.subplots(figsize=SINGLE_PLOT_FIGSIZE)

    bars = ax.barh(columns_with_missing.index, columns_with_missing["missing_pct"],
                    color=ALERT_COLOR)
    ax.set_title(title)
    ax.set_xlabel("Tỷ lệ thiếu (%)")
    ax.set_ylabel("Tên cột")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    # Ghi số % ngay cạnh mỗi thanh để dễ đọc chính xác
    for bar, value in zip(bars, columns_with_missing["missing_pct"]):
        ax.text(value + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%", va="center", fontsize=10)

    plt.tight_layout()

    if save_as is not None:
        save_current_figure(save_as)

    plt.show()

    return missing_summary