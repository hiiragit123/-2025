#!/usr/bin/env python3
"""
セクション4: CSV vs Parquet ファイル形式比較

セクション4の概要:
1. CSV と Parquet の読み書き速度比較
2. ファイルサイズの違い
3. 圧縮効果の比較
"""

import pandas as pd
import polars as pl
import time
import gc
from pathlib import Path
import numpy as np
import os

try:
    from sec_01_setup import get_memory_usage, time_and_memory, format_size
except ImportError:
    print("エラー: sec_01_setup.pyが見つかりません。先にセクション1を実行してください。")
    exit(1)

# =============================================================================
# ファイル作成とサイズ比較
# =============================================================================

def create_and_compare_files(csv_path):
    """
    CSV から Parquet ファイルを作成し、ファイルサイズを比較
    """
    print("=" * 60)
    print("CSV vs Parquet ファイル作成とサイズ比較")
    print("=" * 60)

    data_dir = csv_path.parent
    parquet_path = data_dir / "sample_data.parquet"
    
    # CSV ファイル情報
    csv_size = os.path.getsize(csv_path)
    print(f"元のCSVファイル:")
    print(f"  パス: {csv_path}")
    print(f"  サイズ: {format_size(csv_size)}")
    
    print("\nParquetファイル作成中...")
    
    # pandas で Parquet 作成
    @time_and_memory
    def create_parquet_pandas():
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, index=False)
        return df
    
    df = create_parquet_pandas()
    
    # Parquet ファイル情報
    parquet_size = os.path.getsize(parquet_path)
    compression_ratio = csv_size / parquet_size
    size_reduction = (1 - parquet_size / csv_size) * 100
    
    print(f"\n作成されたParquetファイル:")
    print(f"  パス: {parquet_path}")
    print(f"  サイズ: {format_size(parquet_size)}")
    print(f"  圧縮比: {compression_ratio:.1f}倍")
    print(f"  サイズ削減: {size_reduction:.1f}%")
    
    # 圧縮オプション付きでも作成
    parquet_compressed_path = data_dir / "sample_data_compressed.parquet"
    
    print("\n圧縮Parquetファイル作成中...")
    
    @time_and_memory
    def create_parquet_compressed():
        df.to_parquet(parquet_compressed_path, index=False, compression='snappy')
    
    create_parquet_compressed()
    
    parquet_compressed_size = os.path.getsize(parquet_compressed_path)
    compressed_compression_ratio = csv_size / parquet_compressed_size
    compressed_size_reduction = (1 - parquet_compressed_size / csv_size) * 100
    
    print(f"\n圧縮Parquetファイル (Snappy):")
    print(f"  パス: {parquet_compressed_path}")
    print(f"  サイズ: {format_size(parquet_compressed_size)}")
    print(f"  圧縮比: {compressed_compression_ratio:.1f}倍")
    print(f"  サイズ削減: {compressed_size_reduction:.1f}%")
    
    # ファイル比較サマリー
    print(f"\nファイルサイズ比較サマリー:")
    print(f"  CSV:                {format_size(csv_size):>10s} (基準)")
    print(f"  Parquet:            {format_size(parquet_size):>10s} ({compression_ratio:.1f}x圧縮)")
    print(f"  Parquet (Snappy):   {format_size(parquet_compressed_size):>10s} ({compressed_compression_ratio:.1f}x圧縮)")
    
    del df
    gc.collect()
    
    return {
        'csv_path': csv_path,
        'parquet_path': parquet_path,
        'parquet_compressed_path': parquet_compressed_path,
        'csv_size': csv_size,
        'parquet_size': parquet_size,
        'parquet_compressed_size': parquet_compressed_size
    }

# =============================================================================
# 読み込み速度比較
# =============================================================================

def compare_reading_speed(file_paths):
    """
    CSV と Parquet の読み込み速度を比較
    """
    print("=" * 60)
    print("CSV vs Parquet 読み込み速度比較")
    print("=" * 60)
    
    csv_path = file_paths['csv_path']
    parquet_path = file_paths['parquet_path']
    parquet_compressed_path = file_paths['parquet_compressed_path']
    
    results = {}
    
    # 1. CSV読み込み (pandas)
    print("1. CSV読み込み (pandas)")
    print("-" * 30)
    
    @time_and_memory
    def read_csv_pandas():
        return pd.read_csv(csv_path)
    
    start_memory = get_memory_usage()
    df_csv_pandas = read_csv_pandas()
    csv_pandas_memory = get_memory_usage() - start_memory
    
    print(f"データ形状: {df_csv_pandas.shape}")
    results['csv_pandas'] = {'memory': csv_pandas_memory, 'shape': df_csv_pandas.shape}
    
    del df_csv_pandas
    gc.collect()
    
    # 2. Parquet読み込み (pandas)
    print("\n2. Parquet読み込み (pandas)")
    print("-" * 30)
    
    @time_and_memory
    def read_parquet_pandas():
        return pd.read_parquet(parquet_path)
    
    start_memory = get_memory_usage()
    df_parquet_pandas = read_parquet_pandas()
    parquet_pandas_memory = get_memory_usage() - start_memory
    
    print(f"データ形状: {df_parquet_pandas.shape}")
    results['parquet_pandas'] = {'memory': parquet_pandas_memory, 'shape': df_parquet_pandas.shape}
    
    del df_parquet_pandas
    gc.collect()
    
    # 3. 圧縮Parquet読み込み (pandas)
    print("\n3. 圧縮Parquet読み込み (pandas)")
    print("-" * 30)
    
    @time_and_memory
    def read_parquet_compressed_pandas():
        return pd.read_parquet(parquet_compressed_path)
    
    start_memory = get_memory_usage()
    df_parquet_compressed_pandas = read_parquet_compressed_pandas()
    parquet_compressed_pandas_memory = get_memory_usage() - start_memory
    
    print(f"データ形状: {df_parquet_compressed_pandas.shape}")
    results['parquet_compressed_pandas'] = {'memory': parquet_compressed_pandas_memory, 'shape': df_parquet_compressed_pandas.shape}
    
    del df_parquet_compressed_pandas
    gc.collect()
    
    print("\n" + "="*50 + "\n")
    
    # 4. CSV読み込み (polars)
    print("4. CSV読み込み (polars)")
    print("-" * 30)
    
    @time_and_memory
    def read_csv_polars():
        return pl.read_csv(csv_path)
    
    start_memory = get_memory_usage()
    df_csv_polars = read_csv_polars()
    csv_polars_memory = get_memory_usage() - start_memory
    
    print(f"データ形状: {df_csv_polars.shape}")
    results['csv_polars'] = {'memory': csv_polars_memory, 'shape': df_csv_polars.shape}
    
    del df_csv_polars
    gc.collect()
    
    # 5. Parquet読み込み (polars)
    print("\n5. Parquet読み込み (polars)")
    print("-" * 30)
    
    @time_and_memory
    def read_parquet_polars():
        return pl.read_parquet(parquet_path)
    
    start_memory = get_memory_usage()
    df_parquet_polars = read_parquet_polars()
    parquet_polars_memory = get_memory_usage() - start_memory
    
    print(f"データ形状: {df_parquet_polars.shape}")
    results['parquet_polars'] = {'memory': parquet_polars_memory, 'shape': df_parquet_polars.shape}
    
    del df_parquet_polars
    gc.collect()
    
    # 6. 圧縮Parquet読み込み (polars)
    print("\n6. 圧縮Parquet読み込み (polars)")
    print("-" * 30)
    
    @time_and_memory
    def read_parquet_compressed_polars():
        return pl.read_parquet(parquet_compressed_path)
    
    start_memory = get_memory_usage()
    df_parquet_compressed_polars = read_parquet_compressed_polars()
    parquet_compressed_polars_memory = get_memory_usage() - start_memory
    
    print(f"データ形状: {df_parquet_compressed_polars.shape}")
    results['parquet_compressed_polars'] = {'memory': parquet_compressed_polars_memory, 'shape': df_parquet_compressed_polars.shape}
    
    del df_parquet_compressed_polars
    gc.collect()
    
    return results

# =============================================================================
# 列選択の効率性比較
# =============================================================================

def compare_column_selection(file_paths):
    """
    列選択時のCSV vs Parquetの効率性を比較
    """
    print("=" * 60)
    print("列選択時の効率性比較")
    print("=" * 60)
    
    csv_path = file_paths['csv_path']
    parquet_path = file_paths['parquet_path']
    
    # 一部の列のみ選択
    selected_columns = ['id', 'category', 'price']

    # 全列数の確認
    total_columns = len(pd.read_csv(csv_path, nrows=1).columns)

    print("\n1. CSV + 列選択 (pandas)")
    print("-" * 30)
    
    @time_and_memory
    def csv_column_selection_pandas():
        return pd.read_csv(csv_path, usecols=selected_columns)
    
    df_csv_selected = csv_column_selection_pandas()
    print(f"データ形状: {df_csv_selected.shape}")
    
    del df_csv_selected
    gc.collect()
    
    print("\n2. Parquet + 列選択 (pandas)")
    print("-" * 30)
    
    @time_and_memory
    def parquet_column_selection_pandas():
        return pd.read_parquet(parquet_path, columns=selected_columns)
    
    df_parquet_selected = parquet_column_selection_pandas()
    print(f"データ形状: {df_parquet_selected.shape}")
    
    del df_parquet_selected
    gc.collect()
    
    print("\n3. CSV + 列選択 (polars)")
    print("-" * 30)
    
    @time_and_memory
    def csv_column_selection_polars():
        return pl.read_csv(csv_path).select(selected_columns)
    
    df_csv_selected_polars = csv_column_selection_polars()
    print(f"データ形状: {df_csv_selected_polars.shape}")
    
    del df_csv_selected_polars
    gc.collect()
    
    print("\n4. Parquet + 列選択 (polars)")
    print("-" * 30)
    
    @time_and_memory
    def parquet_column_selection_polars():
        return pl.read_parquet(parquet_path).select(selected_columns)
    
    df_parquet_selected_polars = parquet_column_selection_polars()
    print(f"データ形状: {df_parquet_selected_polars.shape}")
    
    del df_parquet_selected_polars
    gc.collect()
    
    print("\n5. Parquet + 列選択 (polars lazy)")
    print("-" * 30)
    
    @time_and_memory
    def parquet_column_selection_polars_lazy():
        return pl.scan_parquet(parquet_path).select(selected_columns).collect()
    
    df_parquet_selected_lazy = parquet_column_selection_polars_lazy()
    print(f"データ形状: {df_parquet_selected_lazy.shape}")
    
    del df_parquet_selected_lazy
    gc.collect()

# =============================================================================
# 書き込み速度比較
# =============================================================================

def compare_writing_speed(csv_path):
    """
    CSV と Parquet の書き込み速度を比較
    """
    print("=" * 60)
    print("CSV vs Parquet 書き込み速度比較")
    print("=" * 60)
    
    # データ読み込み
    df = pd.read_csv(csv_path)
    data_dir = csv_path.parent
    
    print(f"データ形状: {df.shape}")
    print(f"メモリ使用量: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB")
    
    # 1. CSV書き込み (pandas)
    print("\n1. CSV書き込み (pandas)")
    print("-" * 30)
    
    csv_write_path = data_dir / "test_write.csv"
    
    @time_and_memory
    def write_csv_pandas():
        df.to_csv(csv_write_path, index=False)
    
    write_csv_pandas()
    csv_write_size = os.path.getsize(csv_write_path)
    print(f"ファイルサイズ: {format_size(csv_write_size)}")
    
    # 2. Parquet書き込み (pandas)
    print("\n2. Parquet書き込み (pandas)")
    print("-" * 30)
    
    parquet_write_path = data_dir / "test_write.parquet"
    
    @time_and_memory
    def write_parquet_pandas():
        df.to_parquet(parquet_write_path, index=False)
    
    write_parquet_pandas()
    parquet_write_size = os.path.getsize(parquet_write_path)
    print(f"ファイルサイズ: {format_size(parquet_write_size)}")
    
    # 3. 圧縮Parquet書き込み (pandas)
    print("\n3. 圧縮Parquet書き込み (pandas)")
    print("-" * 30)
    
    parquet_compressed_write_path = data_dir / "test_write_compressed.parquet"
    
    @time_and_memory
    def write_parquet_compressed_pandas():
        df.to_parquet(parquet_compressed_write_path, index=False, compression='snappy')
    
    write_parquet_compressed_pandas()
    parquet_compressed_write_size = os.path.getsize(parquet_compressed_write_path)
    print(f"ファイルサイズ: {format_size(parquet_compressed_write_size)}")
    
    # polarsデータフレームに変換
    df_polars = pl.from_pandas(df)
    
    # 4. CSV書き込み (polars)
    print("\n4. CSV書き込み (polars)")
    print("-" * 30)
    
    csv_write_polars_path = data_dir / "test_write_polars.csv"
    
    @time_and_memory
    def write_csv_polars():
        df_polars.write_csv(csv_write_polars_path)
    
    write_csv_polars()
    csv_write_polars_size = os.path.getsize(csv_write_polars_path)
    print(f"ファイルサイズ: {format_size(csv_write_polars_size)}")
    
    # 5. Parquet書き込み (polars)
    print("\n5. Parquet書き込み (polars)")
    print("-" * 30)
    
    parquet_write_polars_path = data_dir / "test_write_polars.parquet"
    
    @time_and_memory
    def write_parquet_polars():
        df_polars.write_parquet(parquet_write_polars_path)
    
    write_parquet_polars()
    parquet_write_polars_size = os.path.getsize(parquet_write_polars_path)
    print(f"ファイルサイズ: {format_size(parquet_write_polars_size)}")
    
    # サマリー
    print(f"\n書き込み速度比較サマリー:")
    print(f"書き込みファイルサイズ:")
    print(f"  CSV (pandas):           {format_size(csv_write_size):>10s}")
    print(f"  CSV (polars):           {format_size(csv_write_polars_size):>10s}")
    print(f"  Parquet (pandas):       {format_size(parquet_write_size):>10s}")
    print(f"  Parquet (polars):       {format_size(parquet_write_polars_size):>10s}")
    print(f"  Parquet圧縮 (pandas):   {format_size(parquet_compressed_write_size):>10s}")
    
    # クリーンアップ
    for temp_file in [csv_write_path, parquet_write_path, parquet_compressed_write_path, 
                      csv_write_polars_path, parquet_write_polars_path]:
        if temp_file.exists():
            temp_file.unlink()
    
    del df, df_polars
    gc.collect()

# =============================================================================
# メイン実行関数
# =============================================================================

def main():
    """メイン実行関数"""
    print("大きなデータ処理の最適化テクニック")
    print("セクション4: CSV vs Parquet ファイル形式比較")
    print()
    
    # データファイルの確認
    csv_path = Path("data/sample_data.csv")
    if not csv_path.exists():
        print("エラー: サンプルデータが見つかりません。")
        print("先にセクション1を実行してください。")
        return
    
    # 1. ファイル作成とサイズ比較
    file_paths = create_and_compare_files(csv_path)
    
    print("\nメモリクリア中...")
    gc.collect()
    
    # 2. 読み込み速度比較
    reading_results = compare_reading_speed(file_paths)
    
    print("\nメモリクリア中...")
    gc.collect()
    
    # 3. 列選択の効率性比較
    compare_column_selection(file_paths)
    
    print("\nメモリクリア中...")
    gc.collect()
    
    # 4. 書き込み速度比較
    compare_writing_speed(csv_path)
    
    print("\nメモリクリア中...")
    gc.collect()

    return file_paths, reading_results


if __name__ == "__main__":
    file_paths, reading_results = main()
