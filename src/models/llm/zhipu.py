# -*- coding: utf-8 -*-#
import json
from typing import List, Dict, Any, Sequence

import requests
from llama_index.core.base.llms.types import CompletionResponse, ChatResponse, ChatMessage, CompletionResponseAsyncGen, \
    ChatResponseAsyncGen, CompletionResponseGen, ChatResponseGen, LLMMetadata, MessageRole
from llama_index.core.base.query_pipeline.query import CustomQueryComponent
from llama_index.core.llms import LLM

from common.constants import CONFIG_ZHIPU_API
from common.log import get_logger

logger = get_logger()


class ZhipuAILLM(LLM):
    """自定义智谱AI（ChatGLM）的 LLM 模型"""

    def __init__(
            self,
            api_key: str,
            model: str = "GLM-4-Plus",  # 默认模型
            temperature: float = 0.3,  # 法律场景建议低随机性
            top_p: int = 0.9,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        if not self._api_key:
            logger.critical("请先填写智谱AI API_KEY！")
            exit(1)
        self._model = model
        if not self._model:
            logger.critical("请选择智谱大语言模型！")
            exit(1)
        self._temperature = temperature
        self._top_p = top_p
        logger.info(f"使用智谱AI {self._model} 模型作为 LLM 大语言模型")

    def _call_api(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """调用智谱API核心方法"""
        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "top_p": kwargs.get("top_p", self._top_p),
            **kwargs
        }
        try:
            logger.debug(f"请求智谱 LLM 大语言模型[{self._model}]：\nurl: {url}\nheaders: {headers}\ndata: {data}")
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            logger.debug(f"response：\n{json.loads(response.text)}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"智谱 LLM 大语言模型[{self._model}]：API request failed: {str(e)}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"智谱 LLM 大语言模型[{self._model}]：Failed to parse API response")
        return {}

    # --------- 必须实现的抽象方法 ---------
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self._model,
            is_chat_model=True
        )

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        response = self._call_api(messages, **kwargs)
        return CompletionResponse(text=response["choices"][0]["message"]["content"])

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("智谱AI当前版本不支持流式补全")

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        raise NotImplementedError("智谱AI当前版本不支持异步流式补全")

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        chat_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        response = self._call_api(chat_messages, **kwargs)
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response["choices"][0]["message"]["content"]
            )
        )

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        raise NotImplementedError("智谱AI当前版本不支持流式对话")

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        raise NotImplementedError("智谱AI当前版本不支持异步流式对话")

    def _as_query_component(self, **kwargs: Any) -> "QueryComponent":
        return CustomQueryComponent(**kwargs)


if __name__ == '__main__':
    zhipu = ZhipuAILLM(CONFIG_ZHIPU_API["api_key"])
    # 测试complete方法
    completion = zhipu.complete("请解释合同法第52条")
    print(f"Completion:\n{completion.text}")

    # 测试chat方法
    messages = [
        ChatMessage(role=MessageRole.USER, content="什么是不可抗力？"),
        ChatMessage(role=MessageRole.ASSISTANT, content="不可抗力是指不能预见、不能避免且不能克服的客观情况。"),
        ChatMessage(role=MessageRole.USER, content="在合同法中如何应用？")
    ]
    chat_response = zhipu.chat(messages)
    print(f"Chat:\n{chat_response.message.content}")
