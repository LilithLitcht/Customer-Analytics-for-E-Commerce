# Chứa các hàm phục vụ Market Basket Analysis (MBA)
# Chuyển dữ liệu giao dịch sang định dạng One hot, chạy thuật toán Apriori và FP-Growth để tìm frequent itemsets, sinh luật kết hợp kèm các độ đo đánh giá (support, confidence, lift, Kulczynski).

import time
import pandas as pd

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


# Hàm chuyển dữ liệu giao dịch sang dạng ma trận One hot (mỗi dòng là 1 hoá đơn, mỗi cột là 1 sản phẩm, giá trị True/False)
def build_onehot_basket(df: pd.DataFrame,
                          invoice_col: str = "Invoice",
                          item_col: str = "StockCode") -> pd.DataFrame:

    # Gom nhóm theo hoá đơn, mỗi hoá đơn
    basket_list = df.groupby(invoice_col)[item_col].apply(list).tolist()

    print(f"[Build Onehot Basket] Số hoá đơn (giỏ hàng): {len(basket_list):,}")

    # TransactionEncoder chuyển danh sách giỏ hàng thành ma trận nhị phân
    encoder = TransactionEncoder()
    encoded_array = encoder.fit(basket_list).transform(basket_list, sparse=True)

    onehot_df = pd.DataFrame.sparse.from_spmatrix(
        encoded_array, columns=encoder.columns_
    )

    print(f"[Build Onehot Basket] Kích thước ma trận Onehot: {onehot_df.shape}")

    return onehot_df


# Hàm chạy Apriori để tìm các frequent itemsets (tập sản phẩm phổ biến) từ ma trận one-hot
def run_apriori(onehot_df: pd.DataFrame,
                 min_support: float = 0.02,
                 low_memory: bool = True) -> tuple[pd.DataFrame, float]:
    
    start_time = time.time()
    frequent_itemsets = apriori(onehot_df, min_support=min_support, use_colnames=True, low_memory=low_memory,)
    elapsed_seconds = time.time() - start_time

    print(f"[Run Apriori] Min Support = {min_support}: tìm được "
          f"{len(frequent_itemsets):,} Frequent Itemsets trong "
          f"{elapsed_seconds:.2f} giây.")

    return frequent_itemsets, elapsed_seconds


# Hàm chạy FP-Growth để tìm các frequent itemsets (tập sản phẩm phổ biến) từ ma trận one-hot
def run_fpgrowth(onehot_df: pd.DataFrame,
                  min_support: float = 0.02) -> tuple[pd.DataFrame, float]:
    
    start_time = time.time()
    frequent_itemsets = fpgrowth(onehot_df, min_support=min_support, use_colnames=True,)
    elapsed_seconds = time.time() - start_time

    print(f"[Run FP-Growth] Min Support = {min_support}: tìm được "
          f"{len(frequent_itemsets):,} Frequent Itemsets trong "
          f"{elapsed_seconds:.2f} giây.")

    return frequent_itemsets, elapsed_seconds



# Hàm sinh luật kết hợp kèm Kulczynski measure
def generate_rules(frequent_itemsets: pd.DataFrame,
                    n_transactions: int,
                    metric: str = "confidence",
                    min_threshold: float = 0.3) -> pd.DataFrame:

    rules = association_rules(frequent_itemsets, num_itemsets=n_transactions, metric=metric, min_threshold=min_threshold,)

    print(f"[Generate Rules] Sinh được {len(rules):,} luật kết hợp "
          f" (Lọc theo {metric} >= {min_threshold}).")

    return rules


# Hàm diễn giải luật sang tên sản phẩm dễ đọc, dùng để trình bày kết quả cho người dùng cuối
def map_stockcode_to_description(df: pd.DataFrame,
                                   stockcode_col: str = "StockCode",
                                   description_col: str = "Description") -> dict:
    # Với mỗi StockCode, lấy Description xuất hiện nhiều nhất
    mode_description = (
        df.dropna(subset=[description_col])
        .groupby(stockcode_col)[description_col]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Không rõ")
    )

    return mode_description.to_dict()


# Hàm chuyển 1 dòng luật kết hợp thành 1 câu tiếng Việt dễ đọc
def format_rule_as_text(rule_row: pd.Series, code_to_name: dict) -> str:
    antecedent_names = [code_to_name.get(code, code) for code in rule_row["antecedents"]]
    consequent_names = [code_to_name.get(code, code) for code in rule_row["consequents"]]

    antecedent_text = ", ".join(antecedent_names)
    consequent_text = ", ".join(consequent_names)

    return f"Khách mua [{antecedent_text}] thường mua kèm [{consequent_text}]"