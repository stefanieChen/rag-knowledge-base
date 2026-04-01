#!/usr/bin/env python3
"""
ChromaDB 管理工具脚本

用于管理 RAG 系统中 ChromaDB 向量存储的文件，包括：
- 查看当前存储的文件列表
- 删除指定文件的所有 chunks
- 按条件批量删除文件
- 清空整个数据库

使用方法:
    python scripts/manage_chromadb.py list                    # 列出所有文件
    python scripts/manage_chromadb.py delete <file_path>      # 删除特定文件
    python scripts/manage_chromadb.py delete-by-repo <repo>   # 按仓库名删除
    python scripts/manage_chromadb.py clear                    # 清空数据库（危险操作）
    python scripts/manage_chromadb.py stats                    # 显示统计信息

作者: RAG System Administrator
日期: 2026-04-01
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.vector_store import VectorStore
from src.config import load_config
from src.logging.logger import get_logger

logger = get_logger("scripts.manage_chromadb")


def _get_all_files(vector_store: VectorStore) -> Dict[str, Dict]:
    """从 ChromaDB 中获取所有文件的信息
    
    直接查询 collection metadata，提取每个 source_file 的 chunks 数量和其他信息。
    兼容文档 ingestion（无 file_hash）和代码 ingestion（有 file_hash）两种路径。

    Args:
        vector_store: VectorStore 实例

    Returns:
        Dict[source_file -> {"chunk_count": int, "file_hash": str, "content_type": str}]
    """
    total = vector_store.count
    if total == 0:
        return {}

    # 直接从 collection 获取所有 metadata
    result = vector_store._collection.get(
        include=["metadatas"],
        limit=total,
    )

    files_info: Dict[str, Dict] = {}
    if result and result["metadatas"]:
        for meta in result["metadatas"]:
            src = meta.get("source_file", "")
            if not src:
                continue
            if src not in files_info:
                files_info[src] = {
                    "chunk_count": 0,
                    "file_hash": meta.get("file_hash", ""),
                    "content_type": meta.get("content_type", ""),
                    "format": meta.get("format", ""),
                }
            files_info[src]["chunk_count"] += 1

    return files_info


def list_files(vector_store: VectorStore) -> None:
    """列出当前存储在 ChromaDB 中的所有文件
    
    直接扫描所有 chunk 的 metadata，按 source_file 分组统计，
    兼容文档 ingestion（无 file_hash）和代码 ingestion（有 file_hash）。

    Args:
        vector_store: VectorStore 实例
    """
    print("\n" + "="*80)
    print("📁 ChromaDB 中存储的文件列表")
    print("="*80)
    
    files_info = _get_all_files(vector_store)
    
    if not files_info:
        total = vector_store.count
        if total > 0:
            print(f"⚠️  数据库中有 {total} 个 chunks，但 metadata 中没有 source_file 字段")
        else:
            print("📭 数据库中没有存储任何文件")
        return
    
    print(f"📊 总计: {len(files_info)} 个文件")
    print(f"🔢 总 chunks 数: {vector_store.count}")
    print("\n文件详情:")
    print("-" * 80)
    
    # 按文件路径排序显示
    for i, (file_path, info) in enumerate(sorted(files_info.items()), 1):
        # 截断过长的路径
        display_path = file_path if len(file_path) <= 60 else "..." + file_path[-57:]
        print(f"{i:3d}. {display_path}")
        print(f"     Chunks: {info['chunk_count']}  |  类型: {info['content_type']}/{info['format']}")
        if info["file_hash"]:
            print(f"     Hash: {info['file_hash']}")
        print()
    
    print("="*80)


def delete_file(vector_store: VectorStore, file_path: str) -> None:
    """删除指定文件的所有 chunks
    
    Args:
        vector_store: VectorStore 实例
        file_path: 要删除的文件路径
    """
    print(f"\n🗑️  正在删除文件: {file_path}")
    
    # 检查文件是否存在（使用完整 metadata 扫描，兼容无 file_hash 的文档 ingestion）
    files_info = _get_all_files(vector_store)
    if file_path not in files_info:
        print(f"❌ 文件 '{file_path}' 在数据库中不存在")
        print("\n💡 可用文件:")
        for available_file in sorted(files_info.keys()):
            print(f"   - {available_file}")
        return
    
    # 确认删除
    confirm = input(f"⚠️  确定要删除文件 '{file_path}' 吗？这将删除该文件的所有 chunks (y/N): ")
    if confirm.lower() != 'y':
        print("❌ 操作已取消")
        return
    
    # 执行删除
    try:
        deleted_count = vector_store.delete_by_source_file(file_path)
        print(f"✅ 成功删除 {deleted_count} 个 chunks")
        print(f"📊 剩余 chunks 数: {vector_store.count}")
    except Exception as e:
        print(f"❌ 删除失败: {e}")


def delete_by_repo(vector_store: VectorStore, repo_name: str) -> None:
    """按仓库名删除所有相关文件
    
    Args:
        vector_store: VectorStore 实例
        repo_name: 仓库名称
    """
    print(f"\n🗑️  正在删除仓库 '{repo_name}' 的所有文件")
    
    # 查找匹配的文件（使用完整 metadata 扫描，兼容无 file_hash 的文档 ingestion）
    files_info = _get_all_files(vector_store)
    matching_files = [f for f in files_info.keys() if repo_name in f]
    
    if not matching_files:
        print(f"❌ 没有找到包含 '{repo_name}' 的文件")
        return
    
    print(f"📋 找到 {len(matching_files)} 个匹配文件:")
    for file_path in matching_files:
        print(f"   - {file_path}")
    
    # 确认删除
    confirm = input(f"\n⚠️  确定要删除这 {len(matching_files)} 个文件吗？(y/N): ")
    if confirm.lower() != 'y':
        print("❌ 操作已取消")
        return
    
    # 执行删除
    total_deleted = 0
    for file_path in matching_files:
        try:
            deleted_count = vector_store.delete_by_source_file(file_path)
            total_deleted += deleted_count
            print(f"✅ 删除 '{file_path}': {deleted_count} chunks")
        except Exception as e:
            print(f"❌ 删除 '{file_path}' 失败: {e}")
    
    print(f"\n📊 总计删除 {total_deleted} 个 chunks")
    print(f"📊 剩余 chunks 数: {vector_store.count}")


def delete_by_metadata(vector_store: VectorStore, metadata_filter: str) -> None:
    """按元数据条件删除文件
    
    Args:
        vector_store: VectorStore 实例
        metadata_filter: 元数据过滤条件，格式为 "key:value"
    """
    try:
        key, value = metadata_filter.split(":", 1)
        where_clause = {key: value}
    except ValueError:
        print("❌ 元数据格式错误，应为 'key:value'")
        return
    
    print(f"\n🗑️  正在删除匹配元数据 {where_clause} 的文件")
    
    # 确认删除
    confirm = input(f"⚠️  确定要删除所有匹配 {where_clause} 的 chunks 吗？(y/N): ")
    if confirm.lower() != 'y':
        print("❌ 操作已取消")
        return
    
    try:
        deleted_count = vector_store.delete_by_metadata(where_clause)
        print(f"✅ 成功删除 {deleted_count} 个 chunks")
        print(f"📊 剩余 chunks 数: {vector_store.count}")
    except Exception as e:
        print(f"❌ 删除失败: {e}")


def show_stats(vector_store: VectorStore) -> None:
    """显示数据库统计信息
    
    Args:
        vector_store: VectorStore 实例
    """
    print("\n" + "="*60)
    print("📊 ChromaDB 统计信息")
    print("="*60)
    
    config = load_config()
    vs_cfg = config.get("vector_store", {})
    
    print(f"📁 数据库路径: {vs_cfg.get('persist_directory', 'N/A')}")
    print(f"🏷️  集合名称: {vector_store.collection_name}")
    print(f"🔢 总 chunks 数: {vector_store.count}")
    
    files_info = _get_all_files(vector_store)
    print(f"📁 总文件数: {len(files_info)}")
    
    if files_info:
        # 统计文件类型
        file_types = {}
        for file_path in files_info.keys():
            ext = Path(file_path).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        print("\n📋 文件类型分布:")
        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            ext_display = ext if ext else "(无扩展名)"
            print(f"   {ext_display}: {count} 个文件")
        
        # 统计内容类型
        content_types = {}
        for info in files_info.values():
            ct = info.get("content_type", "unknown")
            content_types[ct] = content_types.get(ct, 0) + 1
        
        print("\n📋 内容类型分布:")
        for ct, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {ct}: {count} 个文件")
    
    print("="*60)


def clear_database(vector_store: VectorStore) -> None:
    """清空整个数据库（危险操作）
    
    Args:
        vector_store: VectorStore 实例
    """
    print("\n⚠️  ⚠️  ⚠️  警告：危险操作 ⚠️  ⚠️  ⚠️")
    print("此操作将删除数据库中的所有数据，且无法恢复！")
    print(f"当前数据库包含 {vector_store.count} 个 chunks")
    
    # 双重确认
    confirm1 = input("❓ 确定要清空整个数据库吗？输入 'CLEAR-ALL' 确认: ")
    if confirm1 != "CLEAR-ALL":
        print("❌ 操作已取消")
        return
    
    confirm2 = input("❓ 最后确认：真的要删除所有数据吗？(yes/no): ")
    if confirm2.lower() != "yes":
        print("❌ 操作已取消")
        return
    
    try:
        # 获取所有 ID 并删除
        total = vector_store.count
        if total > 0:
            result = vector_store._collection.get(include=[], limit=total)
            if result and result["ids"]:
                vector_store._collection.delete(ids=result["ids"])
                print(f"✅ 成功删除所有 {total} 个 chunks")
            else:
                print("📭 数据库已经是空的")
        else:
            print("📭 数据库已经是空的")
        
        print(f"📊 当前 chunks 数: {vector_store.count}")
        
    except Exception as e:
        print(f"❌ 清空数据库失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ChromaDB 管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s list                           # 列出所有文件
  %(prog)s delete /path/to/file.pdf      # 删除特定文件
  %(prog)s delete-by-repo my-repo         # 按仓库名删除
  %(prog)s delete-by-meta repo_name:test  # 按元数据删除
  %(prog)s stats                          # 显示统计信息
  %(prog)s clear                          # 清空数据库（危险）
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='列出所有存储的文件')
    
    # delete 命令
    delete_parser = subparsers.add_parser('delete', help='删除指定文件')
    delete_parser.add_argument('file_path', help='要删除的文件路径')
    
    # delete-by-repo 命令
    delete_repo_parser = subparsers.add_parser('delete-by-repo', help='按仓库名删除文件')
    delete_repo_parser.add_argument('repo_name', help='仓库名称（支持部分匹配）')
    
    # delete-by-meta 命令
    delete_meta_parser = subparsers.add_parser('delete-by-meta', help='按元数据删除文件')
    delete_meta_parser.add_argument('metadata', help='元数据过滤条件，格式: key:value')
    
    # stats 命令
    stats_parser = subparsers.add_parser('stats', help='显示统计信息')
    
    # clear 命令
    clear_parser = subparsers.add_parser('clear', help='清空整个数据库（危险操作）')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # 初始化向量存储
        vector_store = VectorStore()
        
        # 执行相应命令
        if args.command == 'list':
            list_files(vector_store)
        elif args.command == 'delete':
            delete_file(vector_store, args.file_path)
        elif args.command == 'delete-by-repo':
            delete_by_repo(vector_store, args.repo_name)
        elif args.command == 'delete-by-meta':
            delete_by_metadata(vector_store, args.metadata)
        elif args.command == 'stats':
            show_stats(vector_store)
        elif args.command == 'clear':
            clear_database(vector_store)
        else:
            print(f"❌ 未知命令: {args.command}")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\n❌ 操作被用户中断")
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        logger.error(f"Script execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
