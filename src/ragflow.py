# -*- coding: utf-8 -*-#
from llama_index.core import get_response_synthesizer, Settings

from models.embedding.baidu import BaiduEmbedding
from common.constants import CONFIG_RAG, CONFIG_BAIDU_API, CONFIG_ZHIPU_API, DATA_DIR, CHROMA_DB_DIR, STORAGE_DIR, \
    CHROMA_DB_COLLECTION_NAME
from data_handler import DataHandler
from models.llm.zhipu import ZhipuAILLM
from models.rerank.baidu import BaiduRerankerPostprocessor
from common.decorator import timer, singleton


@singleton
class RagFlow:
    def __init__(self):
        self.embedding_model = BaiduEmbedding(
            api_key=CONFIG_BAIDU_API['api_key'],
            model=CONFIG_BAIDU_API.get("embedding_model")
        )
        self.rerank_model = BaiduRerankerPostprocessor(
            api_key=CONFIG_BAIDU_API['api_key'],
            top_n=CONFIG_RAG['rerank_top_n'],
            model=CONFIG_BAIDU_API.get("rerank_model")
        )
        self.llm_model = ZhipuAILLM(
            api_key=CONFIG_ZHIPU_API['api_key'],
            model=CONFIG_ZHIPU_API.get("llm_model"),
            temperature=CONFIG_RAG['llm_temperature'],
            top_p=CONFIG_RAG['llm_top_p'],
        )
        Settings.embed_model = self.embedding_model
        Settings.llm = self.llm_model
        # 数据读取，并创建存储
        data_handler = DataHandler(
            data_dir=str(DATA_DIR),
            chroma_db_dir=str(CHROMA_DB_DIR),
            persist_dir=str(STORAGE_DIR),
            collection_name=CHROMA_DB_COLLECTION_NAME
        )
        index = data_handler.init_vector_store()

        # 创建检索器和响应合成器
        self.retriever = index.as_retriever(
            similarity_top_k=CONFIG_RAG["top_k"],  # 初始检索数量
            vector_store_query_mode="hybrid",  # 混合检索模式
            alpha=0.5,  # 平衡密集检索与稀疏检索
            # filters={"content_type": "legal_article"}  # 添加元数据过滤
        )
        self.response_synthesizer = get_response_synthesizer(
            # text_qa_template=response_template,
            verbose=True
        )

    @timer
    def retrieve(self, question):
        return self.retriever.retrieve(question)

    @timer
    def rerank(self, question, nodes):
        reranked_nodes = self.rerank_model.postprocess_nodes(
            nodes,
            query_str=question
        )
        # 执行过滤
        filtered_nodes = [node for node in reranked_nodes if node.score > CONFIG_RAG["rerank_min_score"]]
        return filtered_nodes

    @timer
    def synthesize(self, question, nodes):
        return self.response_synthesizer.synthesize(
            question,
            nodes=nodes
        )

    def answer(self, question):
        # 1. 初始检索
        initial_nodes = self.retrieve(question)

        # 2. 重排序
        reranked_nodes = self.rerank(question, initial_nodes)
        if not reranked_nodes:
            response_text = "⚠️ 未找到相关法律条文，请尝试调整问题描述或咨询专业律师。"
        else:
            # 3. 合成答案
            response = self.synthesize(question, reranked_nodes)
            response_text = response.response
        return response_text, reranked_nodes
