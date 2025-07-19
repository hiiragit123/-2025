# 大きなデータ処理の最適化テクニック

このプロジェクトでは、Python における大規模データ処理の最適化手法について実践的に学びます。

## プロジェクト構成

```
subzemi2025/
├── README.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── codes/
│   ├── sec_01_setup.py
│   ├── sec_02_data_type.py
│   ├── sec_03_pd_vs_pl.py
│   └── sec_04_csv_vs_parquet.py
└── data/
    ├── incremental_statistics.json
    ├── large_sample_data.csv
    ├── sample_data.csv
    ├── sample_data.parquet
    └── sample_data_compressed.parquet
```

## 環境構築

### uv を使った高速な環境構築（推奨）

[uv](https://docs.astral.sh/uv/) は Rust で書かれた超高速な Python パッケージマネージャーです。従来の pip や poetry よりも大幅に高速で、依存関係の解決やパッケージのインストールが劇的に早くなります。

#### uv のインストール

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### uv を使った環境構築

```bash
# 1. プロジェクトディレクトリに移動
cd XXX

# 2. 仮想環境の作成とアクティブ化
uv venv
source .venv/bin/activate  # macOS/Linux
# または
.venv\Scripts\activate     # Windows

# 3. 必要なパッケージのインストール
uv sync
# 4. コードの実行
uv run {python ファイルのパス}
```

#### uv の主な利点

- **高速**: pip の 10-100 倍高速なパッケージインストール
- **依存関係解決**: 競合する依存関係を効率的に解決
- **Rust 製**: メモリ安全で高性能
- **pip 互換**: 既存の pip ワークフローをそのまま使用可能

## 各セクションの概要

### セクション 1: 準備とサンプルデータ作成
- パフォーマンス測定ユーティリティ関数
- 実験用データセットの作成
- メモリ使用量監視の基本

### セクション 2: データ型と計算量の最適化
- データ構造の計算量比較（O(n) vs O(1)）
- データ型最適化によるメモリ削減
- 実際のパフォーマンス測定

### セクション 3: pandas vs polars の比較
- 読み込み速度の比較
- lazy evaluation の効果
- メモリ使用量の違い


### セクション 4: CSV vs Parquet ファイル形式比較（追加済み）
- ファイルサイズと圧縮効果
- 読み書き性能の比較
- 列指向データベースの利点

## その他

### メモリプロファイリング

大規模データ処理では、メモリ使用量の分析が必要になります

#### memory_profiler を使った行単位プロファイリング

```python
# インストール
uv pip install memory-profiler

# 使用例
from memory_profiler import profile

@profile
def memory_intensive_function():
    # 大きなリストを作成
    big_list = [i for i in range(1000000)]
    # DataFrame を作成
    df = pd.DataFrame({'data': big_list})
    # 処理
    result = df['data'].sum()
    return result

# 実行時にメモリ使用量を出力
# python -m memory_profiler your_script.py
```

#### psutil を使ったリアルタイムモニタリング

```python
import psutil
import time

def monitor_memory_usage(func, *args, **kwargs):
    """関数実行中のメモリ使用量を監視"""
    process = psutil.Process()
    
    # 実行前のメモリ
    mem_before = process.memory_info().rss / 1024 / 1024
    
    # 関数実行
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # 実行後のメモリ
    mem_after = process.memory_info().rss / 1024 / 1024
    
    print(f"実行時間: {end_time - start_time:.2f}秒")
    print(f"メモリ使用量: {mem_before:.1f}MB → {mem_after:.1f}MB")
    print(f"メモリ増加: +{mem_after - mem_before:.1f}MB")
    
    return result

# 使用例
result = monitor_memory_usage(pd.read_csv, 'large_file.csv')
```

### tqdm を使った進捗管理

大規模データ処理では進捗の可視化が重要です。

#### 基本的な使い方

```python
from tqdm import tqdm
import time

# 基本的なループ
for i in tqdm(range(100), desc="処理中"):
    time.sleep(0.01)  # 何らかの処理

# pandas との組み合わせ
tqdm.pandas(desc="データ処理中")
df['new_column'] = df['old_column'].progress_apply(lambda x: x * 2)
```

#### チャンク処理での活用

```python
from tqdm import tqdm
import pandas as pd

def process_large_file_with_progress(file_path, chunk_size=10000):
    # まずファイルの総行数を取得
    total_rows = sum(1 for _ in open(file_path)) - 1  # ヘッダー除く
    total_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    results = []
    
    # tqdm でチャンク処理の進捗を表示
    with tqdm(total=total_chunks, desc="チャンク処理", unit="chunk") as pbar:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # 何らかの処理
            processed_chunk = chunk.groupby('category').sum()
            results.append(processed_chunk)
            
            # 進捗更新
            pbar.update(1)
            pbar.set_postfix({
                'rows': len(chunk),
                'memory': f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB"
            })
    
    return pd.concat(results)
```

#### 複数の進捗バーを使った詳細表示

```python
from tqdm import tqdm
import time

def nested_progress_example():
    # 外側のループ（ファイル処理）
    files = ['file1.csv', 'file2.csv', 'file3.csv']
    
    for file in tqdm(files, desc="ファイル", position=0):
        # 内側のループ（チャンク処理）
        for chunk_num in tqdm(range(100), desc=f"{file}", position=1, leave=False):
            time.sleep(0.01)  # 処理のシミュレーション
```

#### メモリ使用量も含めた進捗表示

```python
import psutil
from tqdm import tqdm

class MemoryProgressBar:
    def __init__(self, total, desc="処理中"):
        self.pbar = tqdm(total=total, desc=desc)
        self.process = psutil.Process()
    
    def update(self, n=1):
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.pbar.set_postfix({
            'Memory': f"{current_memory:.1f}MB"
        })
        self.pbar.update(n)
    
    def close(self):
        self.pbar.close()

# 使用例
pbar = MemoryProgressBar(total=1000, desc="データ処理")
for i in range(1000):
    # 何らかの処理
    time.sleep(0.001)
    pbar.update(1)
pbar.close()
```

