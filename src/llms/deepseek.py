from openai import OpenAI
from typing import Tuple
import json
from src.config import Config

config = Config()


class DeepSeekClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_API_BASE
        )
        self.messages = []
        self.content = ""
        self.reasoning_content = ""

    def generate(
        self,
        user_prompt: str,
        # 设置的小一点，方便调试
        max_tokens: int = 4096,
    ) -> Tuple[str, str]:
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

    def stream_generate(
        self,
        user_prompt: str,
        max_tokens: int = 4096,
    ):
        """流式返回内容和推理内容"""
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
            else:
                self.content += chunk.choices[0].delta.content

        self.messages.append(
            {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
        )

        return (self.content, self.reasoning_content)

    def view_messages(self):
        """查看对话历史"""
        for message in self.messages:
            print(f"{message['role']}: {message['content']}")
