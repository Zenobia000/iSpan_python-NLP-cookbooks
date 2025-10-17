#!/usr/bin/env python3
"""
iSpan Python NLP Cookbooks v2 - 統一數據下載腳本
Unified Dataset Download Script

執行方式:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --keras
    python scripts/download_datasets.py --huggingface
    python scripts/download_datasets.py --kaggle
    python scripts/download_datasets.py --chinese

作者: iSpan NLP Team
版本: v1.0
日期: 2025-10-17
"""

import os
import sys
import argparse
from pathlib import Path
import requests
import zipfile
import shutil

# 設定項目根目錄
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

def setup_directories():
    """創建必要的目錄結構"""
    print("創建數據集目錄結構...")

    dirs_to_create = [
        DATASETS_DIR,
        DATASETS_DIR / "datasets",
        DATASETS_DIR / "huggingface_cache",
        DATASETS_DIR / "glove",
        DATASETS_DIR / "sms_spam",
        DATASETS_DIR / "twitter_sentiment",
        DATASETS_DIR / "food_delivery",
        DATASETS_DIR / "movielens",
    ]

    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("目錄結構創建完成")


def download_keras_datasets():
    """下載 Keras 內建數據集 (IMDB, Reuters)"""
    print("\n" + "=" * 60)
    print("下載 Keras 內建數據集...")
    print("=" * 60)

    try:
        # 設定 Keras 數據集下載路徑到專案 datasets 目錄
        keras_datasets_dir = DATASETS_DIR / "datasets"
        keras_datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # 移除 KERAS_HOME 設定，讓 Keras 下載到預設路徑
        # os.environ['KERAS_HOME'] = str(keras_datasets_dir.parent)

        from tensorflow import keras
        from pathlib import Path
        import shutil

        # Keras 預設快取路徑
        default_keras_dir = Path.home() / ".keras" / "datasets"

        # 檔案清單
        files_to_move = {
            "imdb.npz": "IMDB 電影評論",
            "reuters.npz": "Reuters 新聞"
        }

        for filename, description in files_to_move.items():
            target_path = keras_datasets_dir / filename
            source_path = default_keras_dir / filename

            if not target_path.exists():
                print(f"下載 {description} 數據集...")
                if "imdb" in filename:
                    keras.datasets.imdb.load_data(num_words=10000)
                elif "reuters" in filename:
                    keras.datasets.reuters.load_data(num_words=10000)
                
                # 從預設路徑移動到目標路徑
                if source_path.exists():
                    shutil.move(str(source_path), str(target_path))
                    print(f"數據集移動完成 -> {target_path}")
                else:
                    print(f"未在預設路徑找到 {filename}")
            else:
                print(f"數據集已存在 -> {target_path}")


        print(f"Keras 數據集快取於: {keras_datasets_dir}")
        print("所有 Keras 數據集下載完成")

    except ImportError:
        print("錯誤: 請先安裝 TensorFlow: pip install tensorflow")
        return False
    except Exception as e:
        print(f"Keras 數據集下載失敗: {e}")
        return False

    return True


def download_huggingface_datasets():
    """下載 Hugging Face 常用數據集"""
    print("\n" + "=" * 60)
    print("下載 Hugging Face 數據集...")
    print("=" * 60)

    try:
        # 設定 Hugging Face 數據集下載路徑到專案 datasets 目錄
        hf_cache_dir = DATASETS_DIR / "huggingface_cache"
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(hf_cache_dir)
        os.environ['HF_DATASETS_CACHE'] = str(hf_cache_dir / "datasets")

        from datasets import load_dataset

        datasets_to_load = [
            ("imdb", None, "IMDB 電影評論"),
            ("ag_news", None, "AG News 新聞分類"),
            # ("conll2003", None, "CoNLL-2003 命名實體識別"), # 改為手動下載
            ("cnn_dailymail", "3.0.0", "CNN/DailyMail 文本摘要"),
            ("squad_v2", None, "SQuAD 2.0 問答系統"),
        ]

        for name, config, description in datasets_to_load:
            try:
                print(f"下載 {description} ({name})...")

                if config:
                    dataset = load_dataset(name, config, split="train[:100]", cache_dir=str(hf_cache_dir / "datasets"))
                else:
                    dataset = load_dataset(name, split="train[:100]", cache_dir=str(hf_cache_dir / "datasets"))

                print(f"{name} 快取完成 (前 100 筆)")

            except Exception as e:
                print(f"{name} 下載失敗: {e}")

        print(f"Hugging Face 數據集快取於: {hf_cache_dir}")
        print("Hugging Face 數據集下載完成")

    except ImportError:
        print("錯誤: 請先安裝 datasets: pip install datasets")
        return False
    except Exception as e:
        print(f"Hugging Face 數據集下載失敗: {e}")
        return False

    return True

def download_conll2003():
    """手動下載 CoNLL-2003 數據集"""
    print("\n" + "=" * 60)
    print("手動下載 CoNLL-2003 數據集...")
    print("=" * 60)

    url = "https://data.deepai.org/conll2003.zip"
    output_dir = DATASETS_DIR / "conll2003"
    zip_path = output_dir / "conll2003.zip"
    train_file = output_dir / "train.txt"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 檢查是否已下載
        if train_file.exists():
            print(f"CoNLL-2003 數據集已存在 -> {output_dir}")
            return True

        print(f"開始下載: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        zip_path.write_bytes(response.content)

        print("解壓縮檔案...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # 刪除 zip 檔案
        zip_path.unlink()

        print(f"CoNLL-2003 數據集下載完成 -> {output_dir}")

    except Exception as e:
        print(f"CoNLL-2003 下載失敗: {e}")
        return False

    return True

def download_glove():
    """下載 GloVe 詞向量"""
    print("\n" + "=" * 60)
    print("下載 GloVe 詞向量 (警告: 檔案較大 ~822MB)...")
    print("=" * 60)

    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    output_dir = DATASETS_DIR / "glove"
    zip_path = output_dir / "glove.6B.zip"

    try:
        print(f"開始下載: {url}")
        print("這可能需要幾分鐘，請耐心等待...")

        # 下載檔案
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # 簡單進度顯示
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r下載進度: {percent:.1f}%", end='')

        print("\n解壓縮檔案...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # 刪除 zip 檔案
        zip_path.unlink()

        print(f"GloVe 詞向量下載完成 -> {output_dir}")

    except Exception as e:
        print(f"GloVe 下載失敗: {e}")
        return False

    return True

def download_kaggle_datasets():
    """下載 Kaggle 數據集 (需要 Kaggle API)"""
    print("\n" + "=" * 60)
    print("下載 Kaggle 數據集...")
    print("=" * 60)

    try:
        import kaggle
    except ImportError:
        print("錯誤: 請先安裝 Kaggle API: pip install kaggle")
        print("並設定 API Token: https://www.kaggle.com/docs/api")
        return False

    datasets_to_download = [
        {
            "name": "uciml/sms-spam-collection-dataset",
            "description": "SMS Spam 垃圾簡訊",
            "output_dir": DATASETS_DIR / "sms_spam"
        },
        {
            "name": "kazanova/sentiment140",
            "description": "Twitter Sentiment 情感分析",
            "output_dir": DATASETS_DIR / "twitter_sentiment"
        },
        {
            "name": "ghoshsaptarshi/av-genpact-hack-dec2018",
            "description": "Food Delivery 外送平台",
            "output_dir": DATASETS_DIR / "food_delivery"
        }
    ]

    for dataset in datasets_to_download:
        try:
            print(f"下載 {dataset['description']} ({dataset['name']})...")

            # 使用 Kaggle API 下載
            kaggle.api.dataset_download_files(
                dataset['name'],
                path=dataset['output_dir'],
                unzip=True
            )

            print(f"{dataset['description']} 下載完成 -> {dataset['output_dir']}")

        except Exception as e:
            print(f"{dataset['description']} 下載失敗: {e}")

    print("Kaggle 數據集下載完成")
    return True

def download_movielens():
    """下載 MovieLens 推薦系統數據集"""
    print("\n" + "=" * 60)
    print("下載 MovieLens 推薦系統數據集...")
    print("=" * 60)

    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    output_dir = DATASETS_DIR / "movielens"
    zip_path = output_dir / "ml-latest-small.zip"

    try:
        print(f"下載 MovieLens 小型數據集: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        zip_path.write_bytes(response.content)

        print("解壓縮檔案...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # 刪除 zip 檔案
        zip_path.unlink()

        print(f"MovieLens 數據集下載完成 -> {output_dir}")

    except Exception as e:
        print(f"MovieLens 下載失敗: {e}")
        return False

    return True

def check_existing_datasets():
    """檢查已存在的數據集"""
    print("\n" + "=" * 60)
    print("檢查現有數據集...")
    print("=" * 60)

    existing_datasets = []

    # 檢查專案自建數據
    project_datasets = [
        ("情歌歌詞", DATASETS_DIR / "lyrics" / "情歌歌詞"),
        ("Google 商家評論", DATASETS_DIR / "google_reviews"),
    ]

    for name, path in project_datasets:
        if path.exists():
            if path.is_dir():
                file_count = len(list(path.glob("*.*")))
                existing_datasets.append(f"{name}: {file_count} 個檔案")
            else:
                existing_datasets.append(f"{name}")
        else:
            existing_datasets.append(f"{name}: 未找到")

    for info in existing_datasets:
        print(info)

    print("=" * 60)

def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="iSpan NLP 課程數據集統一下載工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    python scripts/download_datasets.py --all          # 下載所有數據集
    python scripts/download_datasets.py --keras        # 只下載 Keras 數據集
    python scripts/download_datasets.py --huggingface  # 只下載 Hugging Face 數據集
    python scripts/download_datasets.py --conll2003    # 手動下載 CoNLL-2003
    python scripts/download_datasets.py --chinese      # 只下載中文語料
        """
    )

    parser.add_argument('--all', action='store_true', help='下載所有數據集')
    parser.add_argument('--keras', action='store_true', help='下載 Keras 內建數據集')
    parser.add_argument('--huggingface', action='store_true', help='下載 Hugging Face 數據集')
    parser.add_argument('--conll2003', action='store_true', help='手動下載 CoNLL-2003 數據集')
    parser.add_argument('--glove', action='store_true', help='下載 GloVe 詞向量')
    parser.add_argument('--kaggle', action='store_true', help='下載 Kaggle 數據集')
    parser.add_argument('--movielens', action='store_true', help='下載 MovieLens 數據集')

    args = parser.parse_args()

    # 如果沒有指定任何選項，顯示幫助
    if not any(vars(args).values()):
        parser.print_help()
        return

    # 顯示歡迎訊息
    print("=" * 60)
    print("iSpan Python NLP Cookbooks v2")
    print("   統一數據集下載工具")
    print("=" * 60)
    print(f"數據集目錄: {DATASETS_DIR}")
    print("=" * 60)

    # 創建目錄
    setup_directories()

    # 檢查現有數據集
    check_existing_datasets()

    # 根據參數下載數據集
    success_count = 0
    total_count = 0

    if args.all or args.keras:
        total_count += 1
        if download_keras_datasets():
            success_count += 1

    if args.all or args.huggingface:
        total_count += 1
        if download_huggingface_datasets():
            success_count += 1

    if args.all or args.conll2003:
        total_count += 1
        if download_conll2003():
            success_count += 1

    if args.all or args.glove:
        total_count += 1
        if download_glove():
            success_count += 1

    if args.all or args.kaggle:
        total_count += 1
        if download_kaggle_datasets():
            success_count += 1

    if args.all or args.movielens:
        total_count += 1
        if download_movielens():
            success_count += 1

    # 顯示總結
    print("\n" + "=" * 60)
    print(f"下載完成: {success_count}/{total_count} 成功")
    print("=" * 60)
    print("\n注意事項:")
    print("1. 部分大型數據集 (如 CNN/DailyMail) 僅預先快取前 100 筆，完整數據將在使用時自動下載")
    print("2. Kaggle 數據集需要先設定 API Token")
    print("3. 所有數據集僅用於教學與學術研究目的")
    print("\n數據集使用說明: 請參考 docs/16_wbs_development_plan_template.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
