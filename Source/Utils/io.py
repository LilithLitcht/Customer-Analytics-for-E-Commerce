# Chứa các hàm đọc dữ liệu thô và đọc ghi file CSV dùng chung

from pathlib import Path
import pandas as pd

# Xác định đường dẫn dự án gốc
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Các thư mục con chuẩn hoá
DATA_DIR = PROJECT_ROOT / "Data"
RAW_DATA_DIR = DATA_DIR / "Raw"
INTERIM_DATA_DIR = DATA_DIR / "Interim"
PROCESSED_DATA_DIR = DATA_DIR / "Processed"
FIGURES_DIR = PROJECT_ROOT / "Figures"
MODELS_DIR = PROJECT_ROOT / "Models"

# Tên 2 Sheet trong file Excel gốc
RAW_EXCEL_SHEET_NAMES = ["Year 2009-2010", "Year 2010-2011"]

# Tên file Excel gốc mặc định
RAW_EXCEL_FILENAME = "online_retail_II.xlsx"

# Hàm trả về đương dẫn tuyệt đối của dự án gốc
def get_project_root() -> Path:
    return PROJECT_ROOT


# Hàm đọc file Excel gốc, gộp 2 sheet thành 1 DataFrame duy nhất
def load_raw_excel(source_path: str | Path = None, 
                   sheet_names: list[str] = None) -> pd.DataFrame:
    # Nếu không truyền đường dẫn, dùng đường dẫn chuẩn của dự án
    if source_path is None:
        source_path = RAW_DATA_DIR / RAW_EXCEL_FILENAME
    else:
        source_path = Path(source_path)

    if sheet_names is None:
        sheet_names = RAW_EXCEL_SHEET_NAMES

    # Kiểm tra file có tồn tại không
    if not source_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu gốc tại: {source_path}\n"
        )

    # Gộp 2 Sheet, gắn thêm cột "Sheet" để biết dòng đó thuộc sheet nào
    sheet_dataframes = []
    for sheet_name in sheet_names:
        sheet_df = pd.read_excel(source_path, sheet_name=sheet_name)
        sheet_df["Sheet"] = sheet_name
        sheet_dataframes.append(sheet_df)

    # Đánh lại chỉ số từ 0 cho DataFrame gộp,
    combined_df = pd.concat(sheet_dataframes, ignore_index=True)

    return combined_df


# Hàm lưu DataFrame ra file CSV
def save_to_csv(df: pd.DataFrame,
                 output_path: str | Path,
                 create_parent_dirs: bool = True) -> None:
    output_path = Path(output_path)

    # Tạo folder nếu chưa có
    if create_parent_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Không ghi index để tránh tạo cột thừa khi đọc lại
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Đã lưu file: {output_path}  (Số dòng: {len(df):,} | Số cột: {df.shape[1]})")


# Hàm đọc lại file CSV, trả về DataFrame
def load_csv(source_path: str | Path,
             date_columns: list[str] = None) -> pd.DataFrame:
    source_path = Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file CSV tại: {source_path}\n"
        )

    # Nếu có cột ngày tháng, parse chúng thành datetime
    df = pd.read_csv(source_path, parse_dates=date_columns)

    return df