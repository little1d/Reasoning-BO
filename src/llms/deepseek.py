from openai import OpenAI
from typing import Tuple
from src.config import Config
from src.utils.jsonl import add_to_jsonl

config = Config()


class DeepSeekClient:
    """不同的 generate 方法不能混用！声明一个 client 就只能用一种 generate 方法"""

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
        max_tokens: int = 4096,
        json_output: bool = False,
    ) -> Tuple[str, str]:
        """
        `generate`方法不保留对话记录。这是因为 chat.completions api 本身是 stateless 的
        Reasoning 数据太长，没必要保存对话历史，不如提取关键信息保存到本地
        Single round generation method without retaining message history.
        json_output: 拓展方法，comment_history 要求是 json 格式，但 ds 官方说这个 api 可能会输出为空，建议加 system_prompt 指导。
        目前是在 user_prompt 中加 prompt 规定 json output，可以达到要求。
        如果测试稳定可以用这个 api，减少prompt，加速响应（但不排除 ds 官方也是通过 prompt 来强行适配）。
        """
        self.messages = []
        if json_output:
            system_prompt = "response in JSON format"
            self.messages.append({"role": "system", "content": system_prompt})

        self.messages.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model=config.DEEPSEEK_MODEL_NAME,
            messages=self.messages,
            max_tokens=max_tokens,
            stream=False,
        )
        self.messages.append(
            # think 只在generate 方法中加了
            {
                "role": "think",
                "content": response.choices[0].message.reasoning_content,
            },
        )
        self.messages.append(
            {
                "role": "assistant",
                "content": response.choices[0].message.content,
            },
        ),
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
        Multi-round 是很有必要的，messages 通过 append 方式增量存储数据，并且 user_prompt 和 content
        天然后成 QA 对，再结合reasoning data 就可以 distill 小模型。少 IO，不用从外部再赋值一遍，同时避免了 jsonl 的复杂存取。
        但是可能会超上下文，，且 system_prompt 包含如果 128K（约等于 96000～128000 汉字）
        # TODO
        但其实又没啥必要，多轮对话的 user_prompt（包含 comment_history）太多重复，宁愿牺牲 io 时间换上下文窗口和 llm 处理时间。 我已经想到了新的 distill data 构造方式
        question: 历史 metrics 数据和 arm，为一个表格，再加一些 prompts。 类似：”这是历史的数据和结果，你应该怎么安排来提高 metrics“。
        reasoning: deepseek generate 的 reasoning data， jsonl 格式
        comment: 保存的 comment data ， jsonl 格式
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
            },
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
        """查看对话历史，方便调试"""
        for message in self.messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

    def save_messages(self, file_path):
        """
        Save client.messages(list) content as JSONL format with paired user, think, assistant content.

        Args:
            file_path: The filepath where JSONL data will be saved.
        """

        print("Start saving the message data for this round of trials.\n")
        distill_data = {}
        for message in self.messages:
            role = message.get("role")
            content = message.get("content")

            if role == "user":
                distill_data["user"] = content
            elif role == "think":
                distill_data["think"] = content
            elif role == "assistant":
                distill_data["assistant"] = content
        add_to_jsonl(file_path, distill_data)
        print("Done!\n")
