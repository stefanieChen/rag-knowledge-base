#!/usr/bin/env python3
"""
CLI 脚本：代码仓库 ingestion

用于从命令行 ingest 代码仓库到 ChromaDB，支持本地文件夹和 Git URL。

使用方法:
    python scripts/ingest_code.py /path/to/code/repo
    python scripts/ingest_code.py /path/to/code/repo "my-repo-name"
    python scripts/ingest_code.py https://github.com/user/repo
    python scripts/ingest_code.py https://github.com/user/repo "custom-name"

作者: RAG System Administrator
日期: 2026-04-01
"""

import argparse
import sys
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ingest import run_code_ingestion
from src.logging.logger import get_logger

logger = get_logger("scripts.ingest_code")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CLI 代码仓库 ingestion 工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s /path/to/local/code
  %(prog)s /path/to/local/code "my-project"
  %(prog)s https://github.com/user/repo
  %(prog)s https://github.com/user/repo "custom-repo-name"
        """
    )
    
    parser.add_argument(
        "path",
        help="本地文件夹路径或 Git 仓库 URL"
    )
    
    parser.add_argument(
        "repo_name",
        nargs="?",
        help="仓库名称（可选，默认自动检测）"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger("ingestion").setLevel(logging.DEBUG)
    
    try:
        print(f"🚀 开始 ingest 代码仓库: {args.path}")
        if args.repo_name:
            print(f"📛 仓库名称: {args.repo_name}")
        else:
            print("📛 仓库名称: 自动检测")
        
        start_time = time.perf_counter()
        
        # 调用代码 ingestion
        file_count, chunk_count, added_count = run_code_ingestion(
            path=args.path,
            repo_name=args.repo_name
        )
        
        elapsed = time.perf_counter() - start_time
        
        print(f"\n✅ Ingest 完成!")
        print(f"📊 统计信息:")
        print(f"   - 扫描文件数: {file_count}")
        print(f"   - 处理 chunks: {chunk_count}")
        print(f"   - 新增 chunks: {added_count}")
        print(f"   - 耗时: {elapsed:.1f}s")
        
        if added_count == 0:
            print("\n💡 没有新增 chunks，可能是因为:")
            print("   - 所有文件都已存在且未变更")
            print("   - 没有找到支持的代码文件")
            print("   - 文件内容为空")
        
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断操作")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n❌ 路径不存在: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ 运行时错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        logger.error(f"Script execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
