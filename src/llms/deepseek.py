from openai import OpenAI
from typing import Tuple
from src.config import Config

config = Config()


class DeepSeekClient:
    """不同的 generate 方法不能混用！声明一个 client 就只能用一种方法"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_API_BASE
        )
        self.messages = []
        self.content = ""
        self.reasoning_content = ""

    def generate(
        self, user_prompt: str, max_tokens: int = 4096
    ) -> Tuple[str, str]:
        """
        Reasoning 数据太长，没必要保存对话历史，不如提取关键信息保存到本地
        Single round generation method without retaining message history.
        """
        self.messages.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=self.messages,
            max_tokens=max_tokens,
            stream=False,
        )
        self.messages.append(
            {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
        )
        return (
            response.choices[0].message.content,
            response.choices[0].message.reasoning_content,
        )

    def multi_round_generate(
        self,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> Tuple[str, str]:
        """
        Multi-round generation method with retained message history.
        """
        """返回元组 (content, reasoning_content) 非流式"""
        # 追加用户消息到历史记录
        self.messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=config.DEEPSEEK_MODEL_NAME,
            messages=self.messages,
            stream=False,
            max_tokens=max_tokens,
        )

        self.content = response.choices[0].message.content
        self.reasoning_content = response.choices[0].message.reasoning_content

        self.messages.append(
            {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
        )
        return (self.content, self.reasoning_content)

    def multi_round_stream_generate(
        self,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> Tuple[str, str]:
        """流式返回内容和推理内容，不好从 response 拿信息，用 content/reasoning content 记录"""
        self.messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=config.DEEPSEEK_MODEL_NAME,
            messages=self.messages,
            stream=True,
            max_tokens=max_tokens,
        )

        self.content = ""
        self.reasoning_content = ""

        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                self.reasoning_content += chunk.choices[
                    0
                ].delta.reasoning_content
            elif chunk.choices[0].delta.content:
                self.content += chunk.choices[0].delta.content

        self.messages.append(
            {
                "role": "assistant",
                "content": self.content,
            }
        )

        return (self.content, self.reasoning_content)

    def view_messages(self):
        """查看对话历史"""
        for message in self.messages:
            print(f"{message['role'].capitalize()}: {message['content']}")
