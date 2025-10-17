"""
批次測試所有 Jupyter Notebooks
使用 nbclient 執行 notebooks 並記錄結果
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

def execute_notebook(notebook_path: Path, timeout: int = 180) -> dict:
    """
    執行單個 notebook 並返回結果

    Args:
        notebook_path: notebook 檔案路徑
        timeout: 超時時間(秒)

    Returns:
        dict: 執行結果 {success, error, time}
    """
    result = {
        'path': str(notebook_path),
        'name': notebook_path.name,
        'success': False,
        'error': None,
        'execution_time': 0
    }

    try:
        # 讀取 notebook
        with open(notebook_path, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # 執行 notebook
        client = NotebookClient(nb, timeout=timeout)
        start = datetime.now()

        print(f"  執行: {notebook_path.name}...", end='')
        client.execute()

        execution_time = (datetime.now() - start).total_seconds()
        result['success'] = True
        result['execution_time'] = execution_time
        print(f" OK ({execution_time:.1f}s)")

    except CellExecutionError as e:
        result['error'] = f"Cell execution error: {str(e)}"
        print(f" FAILED")
        print(f"    Error: {str(e)[:200]}")
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
        print(f" ERROR: {str(e)[:100]}")

    return result

def test_chapter(chapter_path: Path) -> list:
    """測試單個章節的所有 notebooks"""
    results = []

    # 找出所有 .ipynb 檔案
    notebooks = sorted(chapter_path.rglob("*.ipynb"))

    # 過濾掉測試輸出檔案
    notebooks = [nb for nb in notebooks if not nb.name.endswith('_tested.ipynb')]

    if not notebooks:
        print(f"  (無 notebooks)")
        return results

    for nb_path in notebooks:
        result = execute_notebook(nb_path)
        results.append(result)

    return results

def main():
    """主測試流程"""
    base_path = Path("D:/python_workspace/project_nlp/iSpan_python-NLP-cookbooks_v2/課程資料")

    print("=" * 70)
    print("iSpan NLP Notebooks 批次測試")
    print("=" * 70)
    print(f"Python 版本: {sys.version}")
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    all_results = []
    chapters = sorted([d for d in base_path.iterdir() if d.is_dir()])

    for chapter in chapters:
        print(f"\n{chapter.name}:")
        chapter_results = test_chapter(chapter)
        all_results.extend(chapter_results)

    # 生成摘要
    print("\n" + "=" * 70)
    print("測試摘要")
    print("=" * 70)

    total = len(all_results)
    success = sum(1 for r in all_results if r['success'])
    failed = total - success

    print(f"總計: {total} notebooks")
    print(f"成功: {success} ({success/total*100:.1f}%)")
    print(f"失敗: {failed} ({failed/total*100:.1f}%)")

    if failed > 0:
        print(f"\n失敗的 notebooks:")
        for r in all_results:
            if not r['success']:
                print(f"  - {r['name']}")
                print(f"    {r['error'][:150]}")

    # 儲存詳細結果
    report_path = base_path.parent / "docs" / "TESTING_REPORT.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_time': datetime.now().isoformat(),
            'python_version': sys.version,
            'summary': {
                'total': total,
                'success': success,
                'failed': failed,
                'success_rate': f"{success/total*100:.1f}%"
            },
            'results': all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n詳細報告已儲存: {report_path}")
    print("=" * 70)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
