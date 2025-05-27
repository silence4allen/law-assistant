import json
import requests
from typing import List
from llama_index.core.base.embeddings.base import BaseEmbedding

from common.constants import CONFIG_BAIDU_API
from common.log import get_logger

logger = get_logger()


class BaiduEmbedding(BaseEmbedding):
    """自定义百度文心 Embedding 模型（兼容 LlamaIndex）"""

    def __init__(
            self,
            api_key: str,
            model: str = "bge-large-zh",  # 模型名
            embed_batch_size: int = 10,  # 批量处理大小
            **kwargs
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        if not self._api_key:
            logger.critical("请先填写百度 API_KEY！")
            exit(1)
        self._model = model
        if not self._model:
            logger.critical("请选择百度向量模型！")
            exit(1)
        self.embed_batch_size = embed_batch_size
        logger.info(f"使用百度 {self._model} 模型作为 embedding 嵌入模型")

    def _get_embedding(self, text: str) -> List[float]:
        if not text:
            logger.error("请输入需要向量化的文本")
            return []
        url = "https://qianfan.baidubce.com/v2/embeddings"

        payload = json.dumps({
            "model": self._model,
            "input": [text],
        }, ensure_ascii=False)
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
        }
        try:
            logger.debug(
                f"请求百度 embedding 嵌入模型[{self._model}]：\nurl: {url}\nheaders: {headers}\npayload: {payload}")
            response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
            response.raise_for_status()
            logger.debug(f"response：\n{json.loads(response.text)}")
            return response.json()['data'][0]["embedding"]
        except requests.exceptions.RequestException as e:
            logger.error(f"百度 embedding 嵌入模型[{self._model}]：API request failed: {str(e)}")
            return []
        except json.JSONDecodeError:
            logger.error(f"百度 embedding 嵌入模型[{self._model}]：Failed to parse API response")
        return []

    # --------- 必须实现的抽象方法 ---------
    def _get_text_embedding(self, text: str) -> List[float]:
        """文本嵌入（用于文档内容）"""
        return self._get_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        """查询嵌入（用于查询语句）"""
        return self._get_embedding(query)  # 百度不区分查询/文档嵌入

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步文本嵌入"""
        return self._get_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步查询嵌入"""
        return self._get_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量文本嵌入"""
        return [self._get_embedding(text) for text in texts]


if __name__ == '__main__':
    embedding_model = BaiduEmbedding(
        api_key=CONFIG_BAIDU_API['api_key'],
        model=CONFIG_BAIDU_API.get("embedding_model")
    )
    test_embedding = embedding_model.get_text_embedding("测试文本")
    print(f"Embedding维度验证：{len(test_embedding)}")
