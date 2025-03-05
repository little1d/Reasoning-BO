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
        self.content = ""
        self.reasoning_content = ""
        self.messages = []

    def generate(
        self,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> Tuple[str, str]:
        """返回元组 (content, reasoning_content)"""
        self.messages.append({"role": "user", "content": user_prompt})
        print(self.messages)
        response = self.client.chat.completions.create(
            model=config.DEEPSEEK_MODEL_NAME,
            messages=self.messages,
            stream=True,
            max_tokens=max_tokens,
        )

        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                self.reasoning_content += chunk.choices[
                    0
                ].delta.reasoning_content
            else:
                self.content += chunk.choices[0].delta.content

            self.messages.append(response.choices[0].message)
        return (self.content, self.reasoning_content)
