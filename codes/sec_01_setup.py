#!/usr/bin/env python3
#
"""
大きなデータ処理の最適化テクニック
セクション1: 準備とサンプルデータ作成

このスクリプトでは、実験用のデータセットを作成し、パフォーマンス測定のためのユーティリティ関数を定義してる。
"""

import pandas as pd
import time
import os
import psutil
from pathlib import Path
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ユーティリティ関数


def get_memory_usage():
    """現在のメモリ使用量をMBで返す"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def time_and_memory(func):
    """関数の実行時間とメモリ使用量を測定するデコレータ"""

    def wrapper(*args, **kwargs):
        start_memory = get_memory_usage()
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = get_memory_usage()

        print(f"実行時間: {end_time - start_time:.2f}秒")
        print(f"メモリ使用量: {end_memory - start_memory:.2f}MB増加")

        return result

    return wrapper


def format_size(bytes_size):
    """バイトサイズを読みやすい形式に変換"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f}TB"


# =============================================================================
# サンプルデータ作成
# =============================================================================


def create_sample_data(n_rows=1_000_000, data_dir="data"):
    """
    実験用の大きなデータセットを作成

    Parameters:
    n_rows: 作成する行数
    data_dir: データを保存するディレクトリ

    Returns:
    tuple: (csv_path, dataframe)
    """
    print("=" * 60)
    print("サンプルデータ作成")
    print("=" * 60)

    # データディレクトリの作成
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    print(f"データ作成中... {n_rows:,}行")

    # サンプルデータの設定
    np.random.seed(42)  # 再現性のため

    # データ生成
    sample_data = {
        "id": range(n_rows),
        "category": np.random.choice(["A", "B", "C", "D", "E"], n_rows),
        "value": np.random.randn(n_rows) * 100,
        "price": np.random.uniform(10, 1000, n_rows),
        "date": pd.date_range("2020-01-01", periods=n_rows, freq="1H"),
        "is_active": np.random.choice([True, False], n_rows),
        "description": [f"商品_{i%10000}" for i in range(n_rows)],
    }

    # データフレーム作成
    df_sample = pd.DataFrame(sample_data)

    # CSVファイルとして保存
    csv_path = data_path / "sample_data.csv"

    print("CSVファイル保存中...")
    start_time = time.time()
    df_sample.to_csv(csv_path, index=False)
    save_time = time.time() - start_time

    # ファイル情報の表示
    file_size = os.path.getsize(csv_path)

    print(f"ファイルパス: {csv_path}")
    print(f"ファイルサイズ: {format_size(file_size)}")
    print(f" 保存時間: {save_time:.2f}秒")

    print("データ型:")
    for col, dtype in df_sample.dtypes.items():
        memory_usage = df_sample[col].memory_usage(deep=True) / 1024 / 1024
        print(f"  {col:12s}: {str(dtype):12s} ({memory_usage:.2f}MB)")

    total_memory = df_sample.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"{'合計':12s}: {'':12s} ({total_memory:.2f}MB)")

    print("サンプルデータ:")
    print(df_sample.head())

    print(" 基本統計:")
    print(df_sample.describe(include="all"))

    return csv_path, df_sample


def main():
    # 初期メモリ使用量
    initial_memory = get_memory_usage()
    print(f"初期メモリ使用量: {initial_memory:.2f}MB")
    print()

    # サンプルデータ作成
    csv_path, df_sample = create_sample_data()

    # メモリ使用量の変化
    final_memory = get_memory_usage()
    print(f"初期: {initial_memory:.2f}MB")
    print(f"最終: {final_memory:.2f}MB")
    print(f"増加: +{final_memory - initial_memory:.2f}MB")
    return csv_path, df_sample


if __name__ == "__main__":
    csv_path, df_sample = main()
