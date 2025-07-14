#!/usr/bin/env python3
"""
大きなデータ処理の最適化テクニック
セクション2: データ型と計算量の最適化

このファイルの概要:
1. データ構造の計算量比較 (list vs set vs dict)
2. データ型の最適化によるメモリ削減
3. 実際のパフォーマンス測定と比較
"""

import pandas as pd
import numpy as np
import time
import gc
from pathlib import Path
from sec_01_setup import get_memory_usage, time_and_memory, format_size

# =============================================================================
# データ構造の計算量比較
# =============================================================================

def compare_data_structures():
    """
    リスト、セット、辞書の検索性能を比較
    O(n) vs O(1) の違いを実際に測定
    """
    print("=" * 60)
    print("データ構造の計算量比較")
    print("=" * 60)

    # テスト用のデータサイズ
    test_sizes = [10_000, 100_000, 1_000_000]

    for n_items in test_sizes:
        print(f"\nデータサイズ: {n_items:,}件")
        print("-" * 40)

        # テストデータの作成
        test_data = list(range(n_items))
        test_set = set(test_data)
        test_dict = {i: f"value_{i}" for i in test_data}

        # 検索対象（最後の方の要素を検索して差を明確にする）
        target = n_items - 10

        print("メモリ使用量:")
        import sys
        list_size = sys.getsizeof(test_data) / 1024
        set_size = sys.getsizeof(test_set) / 1024
        dict_size = sys.getsizeof(test_dict) / 1024

        print(f"  リスト: {list_size:.1f}KB")
        print(f"  セット: {set_size:.1f}KB ({set_size/list_size:.1f}倍)")
        print(f"  辞書:   {dict_size:.1f}KB ({dict_size/list_size:.1f}倍)")

        print("検索性能テスト:")

        print(f"\n  検索対象: {target}")

        # リスト検索 (O(n))
        start = time.time()
        result_list = target in test_data
        time_list = time.time() - start

        # セット検索 (O(1))
        start = time.time()
        result_set = target in test_set
        time_set = time.time() - start

        # 辞書検索 (O(1))
        start = time.time()
        result_dict = target in test_dict
        time_dict = time.time() - start

        # 結果表示
        print(f"    リスト: {time_list:.6f}秒")
        print(f"    セット: {time_set:.6f}秒 ({time_list/max(time_set, 1e-6):.0f}倍高速)")
        print(f"    辞書:   {time_dict:.6f}秒 ({time_list/max(time_dict, 1e-6):.0f}倍高速)")

        # メモリクリア
        del test_data, test_set, test_dict
        gc.collect()

def demonstrate_set_operations():
    """セットの効率的な操作例"""
    print("\n実践例: 大きなデータでの重複チェック")
    print("-" * 40)

    # 大きなデータセットのシミュレーション
    n_items = 100_000
    data_with_duplicates = list(range(n_items)) + list(range(n_items // 2))

    print(f"データサイズ: {len(data_with_duplicates):,}件")

    # 方法1: リストを使った重複チェック（悪い例）
    start_time = time.time()
    duplicates_list = []
    for item in data_with_duplicates:
        if data_with_duplicates.count(item) > 1 and item not in duplicates_list:
            duplicates_list.append(item)
    time_list = time.time() - start_time

    # 方法2: セットを使った重複チェック（良い例）
    start_time = time.time()
    seen = set()
    duplicates_set = set()
    for item in data_with_duplicates:
        if item in seen:
            duplicates_set.add(item)
        else:
            seen.add(item)
    time_set = time.time() - start_time

    print(f"リスト方式: {time_list:.3f}秒 (重複数: {len(duplicates_list)})")
    print(f"セット方式: {time_set:.3f}秒 (重複数: {len(duplicates_set)})")
    print(f"性能向上: {time_list/time_set:.1f}倍高速")

# =============================================================================
# データ型の最適化
# =============================================================================

def compare_data_types(csv_path):
    """
    データ型の最適化によるメモリ使用量の比較
    """
    print("=" * 60)
    print("データ型の最適化")
    print("=" * 60)

    print("改善前: デフォルトのデータ型で読み込み")

    @time_and_memory
    def load_data_default():
        df = pd.read_csv(csv_path)
        return df

    df_default = load_data_default()

    # メモリ使用量の詳細表示
    memory_usage_default = df_default.memory_usage(deep=True)
    total_memory_default = memory_usage_default.sum() / 1024 / 1024

    print(f"メモリ使用量 (デフォルト): {total_memory_default:.2f}MB")
    print("列別メモリ使用量:")
    for col, usage in memory_usage_default.items():
        if col != 'Index':
            dtype = df_default[col].dtype
            usage_mb = usage / 1024 / 1024
            print(f"  {col:12s}: {str(dtype):12s} {usage_mb:8.2f}MB")

    print("データ型:")
    print(df_default.dtypes)

    # メモリクリア
    del df_default
    gc.collect()

    print("\n" + "="*50 + "\n")

    # 改善後：最適化されたデータ型
    print("🟢 改善後: 最適化されたデータ型で読み込み")

    @time_and_memory
    def load_data_optimized():
        # データ型の最適化
        dtype_dict = {
            'id': 'int32',
            'category': 'category',
            'value': 'float32',
            'price': 'float32',
            'is_active': 'bool',
            'description': 'category',
        }

        df = pd.read_csv(csv_path, dtype=dtype_dict, parse_dates=['date'])
        return df

    df_optimized = load_data_optimized()

    # メモリ使用量の詳細表示
    memory_usage_optimized = df_optimized.memory_usage(deep=True)
    total_memory_optimized = memory_usage_optimized.sum() / 1024 / 1024

    print(f"メモリ使用量 (最適化): {total_memory_optimized:.2f}MB")
    print("列別メモリ使用量:")
    for col, usage in memory_usage_optimized.items():
        if col != 'Index':
            dtype = df_optimized[col].dtype
            usage_mb = usage / 1024 / 1024
            print(f"  {col:12s}: {str(dtype):12s} {usage_mb:8.2f}MB")

    print("データ型:")
    print(df_optimized.dtypes)

    # 改善効果の計算
    memory_reduction = total_memory_default - total_memory_optimized
    reduction_percentage = (memory_reduction / total_memory_default) * 100

    print("最適化効果:")
    print(f"メモリ削減量: {memory_reduction:.2f}MB")
    print(f"削減率: {reduction_percentage:.1f}%")
    print(f"圧縮比: {total_memory_default/total_memory_optimized:.1f}倍")

    return df_optimized



def main():
    """メイン実行関数"""
    print("大きなデータ処理の最適化テクニック")
    print("セクション2: データ型と計算量の最適化")
    print()

    # 1. データ構造の計算量比較
    compare_data_structures()

    # セット操作の実践例
    demonstrate_set_operations()

    # 2. データ型の最適化
    csv_path = Path("data/sample_data.csv")
    if not csv_path.exists():
        print("エラー: サンプルデータが見つかりません。")
        print("先にセクション1を実行してください。")
        return

    df_optimized = compare_data_types(csv_path)

    return df_optimized


if __name__ == "__main__":
    df_optimized = main()
