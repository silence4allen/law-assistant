# -*- coding: utf-8 -*-#
import re
import streamlit as st

from common.log import set_default_logger_name, get_logger
from msg import Msg

set_default_logger_name("law_assistant")
from ragflow import RagFlow

logger = get_logger()


def disable_streamlit_watcher():
    """禁用 Streamlit 文件热重载"""

    def _on_script_changed(_):
        return

    # 确保在 Streamlit 运行时初始化后调用
    if hasattr(st, 'runtime') and hasattr(st.runtime, 'get_instance'):
        st.runtime.get_instance()._on_script_changed = _on_script_changed


def set_streamlit_config():
    """配置 Streamlit"""
    st.title("⚖️ 智能劳动法咨询助手")
    st.markdown("欢迎使用劳动法智能咨询系统～请输入您的问题，我们将基于最新劳动法律法规为您解答。")
    # 初始化会话状态
    if "history" not in st.session_state:
        st.session_state.history = []


def show_reference(nodes):
    """展示参考依据"""
    if not nodes:
        return
    with st.expander("查看法律依据"):
        for idx, node in enumerate(nodes, 1):
            meta = node.node.metadata
            st.markdown(f"**[{idx}] {meta['full_title']}**")
            st.caption(f"来源文件：{meta['source_file']} | 法律名称：{meta['law_name']}")
            st.markdown(f"相关度：`{node.score:.4f}`")
            st.info(f"{node.node.text}")


def show_think(title, think_text):
    """展示思维过程"""
    if not think_text:
        return
    with st.expander(title):
        think_contents = ""
        for think_content in think_text:
            formatted_content = think_content.strip().replace("\n", "<br/>")
            item = f'<span style="color: #808080">{formatted_content}</span>'
            think_contents += item

        st.markdown(think_contents, unsafe_allow_html=True)


def show_chat_content(msg: Msg, show_log: bool = True):
    """展示聊天内容"""
    with st.chat_message(msg.role):
        st.markdown(msg.reply_text if msg.reply_text else msg.content)
    if show_log and msg.role == "user":
        logger.info(f"用户提问：{msg.content}")
    if msg.role == "assistant":
        if show_log:
            logger.info(f"助手回复：{msg.reply_text if msg.reply_text else msg.content}")
        # 展示思维过程
        show_think(title="📝 模型思考过程（点击展开）", think_text=msg.think_text)
        # 展示参考依据
        show_reference(msg.reference_nodes)


def init_chat_interface():
    """初始化聊天界面"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        show_chat_content(msg, show_log=False)


def add_msg_to_history(msg: Msg):
    """添加会话消息到历史"""
    st.session_state.messages.append(msg)


def handle_msg(msg: Msg):
    """处理消息"""
    # 展示聊天内容
    show_chat_content(msg)
    # 添加消息到历史
    add_msg_to_history(msg)


def is_legal_question(text: str) -> bool:
    """判断问题是否属于法律咨询"""
    legal_keywords = ["劳动法", "合同", "工资", "工伤", "解除", "赔偿", "用人单位", "劳动者"]
    return any(keyword in text for keyword in legal_keywords)


def run():
    disable_streamlit_watcher()
    set_streamlit_config()
    init_chat_interface()

    logger.debug(f"历史消息: {st.session_state.messages}")
    with st.spinner("正在构建知识库..."):
        ragflow = RagFlow()

    if question := st.chat_input("请输入劳动法相关问题"):
        question = question.strip()
        # 处理用户问题
        handle_msg(Msg(role="user", content=question))

        # 获取回复
        assistant_msg = Msg(
            role="assistant",
            content="对不起，我暂时无法回答劳动法之外的问题哦～"
        )
        if is_legal_question(question):
            # RAG流程获取回复内容
            with st.spinner("正在分析问题，请稍等..."):
                assistant_msg.content, assistant_msg.reference_nodes = ragflow.answer(question)
                assistant_msg.reply_text = re.sub(r'<think>.*?</think>', '', assistant_msg.content,
                                                  flags=re.DOTALL).strip()
                assistant_msg.think_text = re.findall(r'<think>(.*?)</think>', assistant_msg.content, re.DOTALL)
        handle_msg(assistant_msg)

    logger.info("=" * 50)


if __name__ == '__main__':
    run()
