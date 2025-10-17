#!/usr/bin/env python3
"""
環境檢查腳本 - iSpan Python NLP Cookbooks

此腳本會檢查學習環境是否正確設置,包括:
- Python 版本
- 核心套件安裝狀態
- GPU 可用性 (可選)
- Jupyter 環境

使用方法:
    python scripts/check_environment.py
"""

import sys
import platform
import subprocess
from pathlib import Path


class Colors:
    """終端機顏色"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """列印標題"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text):
    """列印成功訊息"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text):
    """列印警告訊息"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text):
    """列印錯誤訊息"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text):
    """列印資訊"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def check_python_version():
    """檢查 Python 版本"""
    print_header("1. Python 版本檢查")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    print(f"Python 版本: {version_str}")
    print(f"執行路徑: {sys.executable}")
    print(f"作業系統: {platform.system()} {platform.release()}")

    if version.major == 3 and version.minor >= 8:
        print_success(f"Python 版本符合要求 (>= 3.8)")
        return True
    else:
        print_error(f"Python 版本過舊,需要 3.8 或更高版本")
        print_info("請升級 Python: https://www.python.org/downloads/")
        return False


def check_package(package_name, import_name=None, min_version=None):
    """
    檢查套件是否已安裝
    Args:
        package_name: 套件名稱 (用於 pip)
        import_name: 導入名稱 (如果與 package_name 不同)
        min_version: 最低版本要求
    Returns:
        bool: 是否安裝且符合版本要求
    """
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Unknown')

        # 版本檢查
        if min_version and version != 'Unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print_warning(f"{package_name:20} {version:12} (需要 >= {min_version})")
                return False

        print_success(f"{package_name:20} {version:12}")
        return True

    except ImportError:
        print_error(f"{package_name:20} {'未安裝':12}")
        return False


def check_core_packages():
    """檢查核心套件"""
    print_header("2. 核心套件檢查")

    packages = {
        # 套件名稱: (導入名稱, 最低版本)
        'numpy': ('numpy', '1.20.0'),
        'pandas': ('pandas', '1.3.0'),
        'matplotlib': ('matplotlib', '3.5.0'),
        'seaborn': ('seaborn', None),
        'jupyter': ('jupyter', None),
        'notebook': ('notebook', None),
    }

    results = {}
    for pkg, (import_name, min_ver) in packages.items():
        results[pkg] = check_package(pkg, import_name, min_ver)

    installed_count = sum(results.values())
    total_count = len(results)

    print(f"\n已安裝: {installed_count}/{total_count}")

    if installed_count == total_count:
        print_success("所有核心套件已安裝")
        return True
    else:
        print_warning(f"有 {total_count - installed_count} 個套件未安裝")
        print_info("安裝指令: pip install -r requirements.txt")
        return False


def check_nlp_packages():
    """檢查 NLP 套件"""
    print_header("3. NLP 套件檢查")

    packages = {
        'nltk': ('nltk', None),
        'jieba': ('jieba', None),
        'scikit-learn': ('sklearn', None),
    }

    results = {}
    for pkg, (import_name, min_ver) in packages.items():
        results[pkg] = check_package(pkg, import_name, min_ver)

    installed_count = sum(results.values())
    total_count = len(results)

    print(f"\n已安裝: {installed_count}/{total_count}")

    if installed_count == total_count:
        print_success("所有 NLP 套件已安裝")
    else:
        print_warning(f"有 {total_count - installed_count} 個套件未安裝 (可選)")

    return True


def check_deep_learning_packages():
    """檢查深度學習套件"""
    print_header("4. 深度學習套件檢查 (可選)")

    packages = {
        'torch': ('torch', None),
        'transformers': ('transformers', None),
        'datasets': ('datasets', None),
    }

    results = {}
    for pkg, (import_name, min_ver) in packages.items():
        results[pkg] = check_package(pkg, import_name, min_ver)

    # 檢查 GPU
    if results.get('torch', False):
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print_success(f"GPU 可用: {gpu_name}")
            else:
                print_info("GPU 不可用 (將使用 CPU)")
        except Exception as e:
            print_warning(f"GPU 檢查失敗: {e}")

    installed_count = sum(results.values())
    total_count = len(results)

    print(f"\n已安裝: {installed_count}/{total_count}")

    if installed_count > 0:
        print_success(f"深度學習套件部分可用 ({installed_count}/{total_count})")
    else:
        print_info("深度學習套件未安裝 (CH05-CH08 需要)")
        print_info("安裝指令: pip install torch transformers datasets")

    return True


def check_jupyter_environment():
    """檢查 Jupyter 環境"""
    print_header("5. Jupyter 環境檢查")

    # 檢查 Jupyter 可執行性
    try:
        result = subprocess.run(
            ['jupyter', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print_success("Jupyter 可執行")
            print(f"版本資訊:\n{result.stdout}")
            return True
        else:
            print_error("Jupyter 無法執行")
            return False
    except FileNotFoundError:
        print_error("找不到 Jupyter 指令")
        print_info("安裝指令: pip install jupyter notebook")
        return False
    except subprocess.TimeoutExpired:
        print_warning("Jupyter 檢查超時")
        return False


def check_project_structure():
    """檢查專案結構"""
    print_header("6. 專案結構檢查")

    required_dirs = [
        '課程資料',
        'docs',
        'scripts',
    ]

    required_files = [
        'README.md',
        'QUICKSTART.md',
        'requirements.txt',
    ]

    all_ok = True

    # 檢查目錄
    print("目錄檢查:")
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print_success(f"  {dir_name}/")
        else:
            print_error(f"  {dir_name}/ (不存在)")
            all_ok = False

    # 檢查檔案
    print("\n檔案檢查:")
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists() and file_path.is_file():
            print_success(f"  {file_name}")
        else:
            print_warning(f"  {file_name} (不存在)")

    return all_ok


def generate_report(results):
    """生成檢查報告"""
    print_header("檢查報告總結")

    total_checks = len(results)
    passed_checks = sum(results.values())
    pass_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0

    print(f"檢查項目: {passed_checks}/{total_checks} 通過")
    print(f"通過率: {pass_rate:.1f}%\n")

    if pass_rate == 100:
        print_success("🎉 恭喜! 環境設置完美!")
        print_success("你已準備好開始學習 NLP 了!")
        print_info("\n下一步: 執行 'jupyter notebook' 開始學習")
    elif pass_rate >= 80:
        print_success("✓ 環境基本符合要求")
        print_info("可以開始學習基礎章節 (CH01-CH04)")
        print_warning("建議安裝缺少的套件以完成進階章節")
    elif pass_rate >= 50:
        print_warning("⚠ 環境需要改善")
        print_warning("請安裝缺少的核心套件")
        print_info("執行: pip install -r requirements.txt")
    else:
        print_error("✗ 環境設置不完整")
        print_error("請檢查上述錯誤並安裝必要套件")
        print_info("\n參考文檔: QUICKSTART.md")


def main():
    """主函數"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     iSpan Python NLP Cookbooks - 環境檢查工具 v1.0             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")

    results = {
        'Python 版本': check_python_version(),
        '核心套件': check_core_packages(),
        'NLP 套件': check_nlp_packages(),
        '深度學習套件': check_deep_learning_packages(),
        'Jupyter 環境': check_jupyter_environment(),
        '專案結構': check_project_structure(),
    }

    generate_report(results)

    # 返回狀態碼
    if all(results.values()):
        sys.exit(0)  # 成功
    else:
        sys.exit(1)  # 有錯誤


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}檢查已取消{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}發生錯誤: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
