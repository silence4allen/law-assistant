# -*- coding: utf-8 -*-#
import json
from pathlib import Path
from typing import List, Dict

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from common.decorator import timer
from common.log import get_logger

logger = get_logger()


class DataHandler:
    def __init__(self, data_dir: str, chroma_db_dir: str, persist_dir: str, collection_name: str):
        self.data_dir = data_dir
        self.chroma_db_dir = chroma_db_dir
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.json_files = list(Path(self.data_dir).glob("*.json"))
        if not self.json_files:
            logger.error("没有找到json数据，请检查！")

    def _validate_json_files(self) -> List[Dict]:
        """加载并验证JSON文件"""
        all_data = []
        for json_file in self.json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # 验证数据结构
                    if not isinstance(data, list):
                        logger.error(f"文件 {json_file.name} 根元素应为列表")
                        return all_data
                    for item in data:
                        if not isinstance(item, dict):
                            logger.error(f"文件 {json_file.name} 包含非字典元素")
                            return all_data
                        for k, v in item.items():
                            if not isinstance(v, str):
                                logger.error(f"文件 {json_file.name} 中键 '{k}' 的值不是字符串")
                                return all_data
                    all_data.extend({"content": item, "metadata": {"source": json_file.name}} for item in data)
                except Exception as e:
                    logger.error(f"加载文件 {json_file} 失败: {str(e)}")
                    return all_data
        logger.info(f"已加载 {len(all_data)} 个条目")
        return all_data

    def _create_nodes(self) -> List[TextNode]:
        """添加ID稳定性保障"""
        raw_data = self._validate_json_files()
        nodes = []
        for entry in raw_data:
            law_dict = entry["content"]
            source_file = entry["metadata"]["source"]
            for full_title, content in law_dict.items():
                # 生成稳定ID（避免重复）
                node_id = f"{source_file}::{full_title}"
                parts = full_title.split(" ", 1)
                law_name = parts[0] if len(parts) > 0 else "未知法律"
                article = parts[1] if len(parts) > 1 else "未知条款"
                node = TextNode(
                    text=content,
                    id_=node_id,  # 显式设置稳定ID
                    metadata={
                        "law_name": law_name,
                        "article": article,
                        "full_title": full_title,
                        "source_file": source_file,
                        "content_type": "legal_article"
                    }
                )
                nodes.append(node)
        if nodes:
            logger.info(f"生成 {len(nodes)} 个文本节点（ID示例：{nodes[0].id_}）")
        return nodes

    @timer
    def init_vector_store(self) -> VectorStoreIndex:
        nodes = []
        if not Path(self.chroma_db_dir).exists():
            nodes = self._create_nodes()
            if not nodes:
                logger.error("nodes 创建失败！")
                exit(1)

        chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        chroma_collection = chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # 确保存储上下文正确初始化
        storage_context = StorageContext.from_defaults(
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )

        # 判断是否需要新建索引
        if nodes and chroma_collection.count() == 0:
            logger.info(f"创建新索引（{len(nodes)}）个节点）...")
            # 显式将节点添加到存储上下文
            storage_context.docstore.add_documents(nodes)
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                show_progress=True
            )
            # 双重持久化保障
            storage_context.persist(persist_dir=self.persist_dir)
            index.storage_context.persist(persist_dir=self.persist_dir)
        else:
            logger.info("加载已有索引...")
            storage_context = StorageContext.from_defaults(
                persist_dir=self.persist_dir,
                vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
            )
            index = VectorStoreIndex.from_vector_store(
                storage_context.vector_store,
                storage_context=storage_context,
                embed_model=Settings.embed_model
            )

        doc_count = len(storage_context.docstore.docs)
        logger.info(f"DocStore记录数：{doc_count}")

        if doc_count > 0:
            sample_key = next(iter(storage_context.docstore.docs.keys()))
            logger.info(f"示例节点ID：{sample_key}")
        else:
            logger.warning("警告：文档存储为空，请检查节点添加逻辑！")
        return index
