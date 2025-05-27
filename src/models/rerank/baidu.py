import json
import requests
from typing import List, Optional
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field

from common.constants import CONFIG_BAIDU_API
from common.log import get_logger

logger = get_logger()


class BaiduRerankerPostprocessor(BaseNodePostprocessor):
    """百度文心 Rerank 模型实现（兼容 LlamaIndex）"""
    top_n: int = Field(default=3, description="返回的重排序结果数量")

    def __init__(
            self,
            api_key: str,
            top_n: int = 3,
            model: str = "bce_reranker_base",
            **kwargs
    ):
        super().__init__(top_n=top_n, **kwargs)
        self._api_key = api_key
        if not self._api_key:
            logger.critical("请先填写百度 API_KEY！")
            exit(1)
        self._model = model
        if not self._model:
            logger.critical("请选择百度重排序模型！")
            exit(1)
        logger.info(f"使用百度 {self._model} 模型作为 rerank 重排序模型")

    def _call_baidu_rerank(self, query: str, documents: List[str]) -> List[float]:
        """调用百度 Rerank API"""
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/reranker/{self._model}"

        payload = json.dumps({
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }, ensure_ascii=False)
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
        }
        try:
            logger.debug(
                f"请求百度 rerank 重排序模型[{self._model}]：\nurl: {url}\nheaders: {headers}\npayload: {payload}")
            response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
            response.raise_for_status()
            logger.debug(f"response：\n{json.loads(response.text)}")
            return response.json().get("results", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"百度 rerank 重排序模型[{self._model}]：API request failed: {str(e)}")
            return []
        except json.JSONDecodeError:
            logger.error(f"百度 rerank 重排序模型[{self._model}]：Failed to parse API response")
        return []

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[
        NodeWithScore]:
        """重排序核心逻辑"""
        if query_bundle is None or len(nodes) == 0:
            return nodes[:self.top_n]

        # 提取文本和原始节点映射
        node_texts = [node.text for node in nodes]
        # 调用百度Rerank API
        try:
            rerank_results = self._call_baidu_rerank(query_bundle.query_str, node_texts)
            rerank_results.sort(key=lambda x: x.get("index"), reverse=False)
            # 更新节点分数
            for node, rerank_result in zip(nodes, rerank_results):
                node.score = float(rerank_result.get("relevance_score"))

            # 按分数降序排序
            nodes.sort(key=lambda x: x.score, reverse=True)

            # 返回结果
            return nodes[:self.top_n]
        except Exception as e:
            logger.error(f"重排序失败，返回原始结果。错误原因：{str(e)}")
            return nodes[:self.top_n]

    def postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
            query_str: Optional[str] = None,
    ) -> List[NodeWithScore]:
        return self._postprocess_nodes(nodes, query_bundle=QueryBundle(query_str))

    @classmethod
    def class_name(cls) -> str:
        """返回类名标识"""
        return "BaiduRerankerPostprocessor"


if __name__ == '__main__':
    # 测试用例
    rerank_model = BaiduRerankerPostprocessor(
        api_key=CONFIG_BAIDU_API['api_key'],
        model=CONFIG_BAIDU_API.get("rerank_model", "bce_reranker_base")
    )
    result = rerank_model._call_baidu_rerank(
        query="劳动合同是什么？",
        documents=["劳动合同是用人单位与劳动者之间确立劳动关系的协议。", "猪八戒娶媳妇"]
    )
    print(result)
