#!/usr/bin/env python3
"""
大きなデータ処理の最適化テクニック
セクション3: pandas vs polars の比較

セクション3の概要:
1. pandas と polars の読み込み速度比較
2. lazy evaluation (scan_csv) の効果
3. データ処理性能の比較
4. メモリ使用量の違い
5. 実践的な使い分けの指針
"""

import pandas as pd
import polars as pl
import time
import gc
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    from sec_01_setup import get_memory_usage, time_and_memory, format_size
except ImportError:
    print("  section_01_setup.pyが見つかりません。先にセクション1を実行してください。")
    exit(1)

# =============================================================================
# 読み込み速度の比較
# =============================================================================

def compare_reading_performance(csv_path):
    """
    pandas と polars の読み込み速度を比較
    """
    print("=" * 60)
    print("pandas vs polars 読み込み速度比較")
    print("=" * 60)

    file_size = csv_path.stat().st_size
    print(f"ファイル: {csv_path}")
    print(f"サイズ: {format_size(file_size)}")
    print()

    results = {}

    # 1. pandas read_csv
    print("pandas read_csv")
    print("-" * 30)

    @time_and_memory
    def pandas_read_csv():
        return pd.read_csv(csv_path)

    start_memory = get_memory_usage()
    df_pandas = pandas_read_csv()
    pandas_memory = get_memory_usage() - start_memory

    print(f"データ形状: {df_pandas.shape}")
    print(f"データ型数: {len(df_pandas.dtypes.unique())}")

    results['pandas_read'] = {
        'memory': pandas_memory,
        'shape': df_pandas.shape,
        'data_memory': df_pandas.memory_usage(deep=True).sum() / 1024 / 1024
    }

    # メモリクリア
    del df_pandas
    gc.collect()

    print("\n" + "="*50 + "\n")

    # 2. polars read_csv
    print("polars read_csv")
    print("-" * 30)

    @time_and_memory
    def polars_read_csv():
        return pl.read_csv(csv_path)

    start_memory = get_memory_usage()
    df_polars = polars_read_csv()
    polars_memory = get_memory_usage() - start_memory

    print(f"データ形状: {df_polars.shape}")
    print(f"データ型数: {len(df_polars.dtypes)}")

    results['polars_read'] = {
        'memory': polars_memory,
        'shape': df_polars.shape,
        'data_memory': df_polars.estimated_size('mb')
    }

    # メモリクリア
    del df_polars
    gc.collect()

    print("\n" + "="*50 + "\n")

    # 3. polars scan_csv (lazy evaluation)
    print("polars scan_csv (lazy evaluation)")
    print("-" * 30)

    @time_and_memory
    def polars_scan_csv():
        return pl.scan_csv(csv_path)

    start_memory = get_memory_usage()
    ldf_polars = polars_scan_csv()
    scan_memory = get_memory_usage() - start_memory

    print("LazyFrame作成完了")
    print("メモリ使用量: ほぼ0 (lazy evaluationのため)")
    print("※ 実際のデータ読み込みは処理実行時に行われます")

    results['polars_scan'] = {
        'memory': scan_memory,
        'lazy': True
    }

    # lazy evaluationの実行例
    print("Lazy評価の実行例:")

    @time_and_memory
    def execute_lazy_operation():
        return ldf_polars.select(['category', 'price']).collect()

    result_lazy = execute_lazy_operation()
    print(f"選択した列のみ読み込み: {result_lazy.shape}")
    print(f"メモリ使用量: {result_lazy.estimated_size('mb'):.2f}MB")

    return results, ldf_polars

def demonstrate_lazy_evaluation(csv_path):
    """
    lazy evaluation の利点を実演
    """
    print("=" * 60)
    print("Lazy Evaluation の利点")
    print("=" * 60)

    # Eager evaluation (全データ読み込み後に選択)
    print("Eager Evaluation (pandas風)")
    print("-" * 40)

    @time_and_memory
    def eager_approach():
        # 全データを読み込み
        df = pl.read_csv(csv_path)
        # 必要な列だけ選択
        result = df.select(['category', 'price'])
        # カテゴリ別統計
        return result.group_by('category').agg([
            pl.col('price').mean().alias('avg_price'),
            pl.col('price').count().alias('count'),
            pl.col('price').std().alias('std_price')
        ])

    result_eager = eager_approach()
    print("結果:")
    print(result_eager)

    print("\n" + "="*50 + "\n")

    # Lazy evaluation (必要な部分のみ読み込み)
    print("🟢 Lazy Evaluation (polars)")
    print("-" * 40)

    @time_and_memory
    def lazy_approach():
        # クエリを定義（まだ実行されない）
        return (
            pl.scan_csv(csv_path)
            .select(['category', 'price'])  # 必要な列のみ
            .group_by('category')
            .agg([
                pl.col('price').mean().alias('avg_price'),
                pl.col('price').count().alias('count'),
                pl.col('price').std().alias('std_price')
            ])
            .collect()  # ここで初めて実行
        )

    result_lazy = lazy_approach()
    print("結果:")
    print(result_lazy)

    # 結果の比較
    print(f"Lazy Evaluationの利点:")
    print("1.メモリ使用量が少ない（必要な列のみ読み込み）")
    print("2.不要な列を読み込まない")
    print("3.処理の最適化が自動で行われる")
    print("4.大きなファイルでも高速処理")


def benchmark_comprehensive_comparison(csv_path):
    """
    包括的なベンチマーク比較
    """
    print("=" * 60)
    print("包括的なベンチマーク比較")
    print("=" * 60)

    # テストケースの定義
    test_cases = [
        {
            'name': '基本的な読み込み',
            'pandas_func': lambda: pd.read_csv(csv_path),
            'polars_func': lambda: pl.read_csv(csv_path),
            'polars_lazy_func': lambda: pl.scan_csv(csv_path).collect()
        },
        {
            'name': '列選択 + 読み込み',
            'pandas_func': lambda: pd.read_csv(csv_path, usecols=['category', 'price', 'value']),
            'polars_func': lambda: pl.read_csv(csv_path).select(['category', 'price', 'value']),
            'polars_lazy_func': lambda: pl.scan_csv(csv_path).select(['category', 'price', 'value']).collect()
        }
    ]

    results = []

    for test_case in test_cases:
        print(f"\n🧪 テスト: {test_case['name']}")
        print("-" * 40)

        test_result = {'name': test_case['name']}

        # pandas
        start_time = time.time()
        start_memory = get_memory_usage()
        df_pandas = test_case['pandas_func']()
        pandas_time = time.time() - start_time
        pandas_memory = get_memory_usage() - start_memory
        del df_pandas
        gc.collect()

        # polars eager
        start_time = time.time()
        start_memory = get_memory_usage()
        df_polars = test_case['polars_func']()
        polars_time = time.time() - start_time
        polars_memory = get_memory_usage() - start_memory
        del df_polars
        gc.collect()

        # polars lazy
        start_time = time.time()
        start_memory = get_memory_usage()
        df_polars_lazy = test_case['polars_lazy_func']()
        polars_lazy_time = time.time() - start_time
        polars_lazy_memory = get_memory_usage() - start_memory
        del df_polars_lazy
        gc.collect()

        # 結果記録
        test_result.update({
            'pandas_time': pandas_time,
            'polars_time': polars_time,
            'polars_lazy_time': polars_lazy_time,
            'pandas_memory': pandas_memory,
            'polars_memory': polars_memory,
            'polars_lazy_memory': polars_lazy_memory
        })

        results.append(test_result)

        # 結果表示
        print(f"pandas:      {pandas_time:.3f}秒, {pandas_memory:.1f}MB")
        print(f"polars:      {polars_time:.3f}秒, {polars_memory:.1f}MB ({pandas_time/polars_time:.1f}x)")
        print(f"polars lazy: {polars_lazy_time:.3f}秒, {polars_lazy_memory:.1f}MB ({pandas_time/polars_lazy_time:.1f}x)")

    # 総合結果
    print(f"\n総合ベンチマーク結果")
    print("=" * 60)

    avg_pandas_speedup = np.mean([r['pandas_time']/r['polars_time'] for r in results])
    avg_lazy_speedup = np.mean([r['pandas_time']/r['polars_lazy_time'] for r in results])

    print(f"平均性能向上:")
    print(f"  polars eager: {avg_pandas_speedup:.1f}倍高速")
    print(f"  polars lazy:  {avg_lazy_speedup:.1f}倍高速")

    return results


def main():
    """メイン実行関数"""
    print("大きなデータ処理の最適化テクニック")
    print("セクション3: pandas vs polars の比較")
    print()

    # データファイルの確認
    csv_path = Path("data/sample_data.csv")
    if not csv_path.exists():
        print("❌ エラー: サンプルデータが見つかりません。")
        print("先にセクション1を実行してください。")
        return

    # 1. 読み込み速度の比較
    results, ldf = compare_reading_performance(csv_path)

    print("\n" + " メモリクリア中..." + "\n")
    gc.collect()

    # 2. lazy evaluationの実演
    demonstrate_lazy_evaluation(csv_path)

    print("\n" + " メモリクリア中..." + "\n")
    gc.collect()

    # 3. 包括的なベンチマーク
    benchmark_results = benchmark_comprehensive_comparison(csv_path)

    return benchmark_results


if __name__ == "__main__":
    benchmark_results = main()
