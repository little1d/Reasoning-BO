#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-05 21:28:03
@File: src/utils/jsonl.py
@IDE: vscode
@Description:
    封装一些 jsonl 存储与拼接读取的函数
"""


import json


def add_to_jsonl(file_path, data):
    """data 应该要是 python dict 格式!!!!!"""
    # 如果是 json，转换为 python dict
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            raise ValueError("Input string is not a valid JSON object")
    # 将python dict转换为 json 存储
    with open(file_path, 'a+', encoding="utf-8") as file:
        json_line = json.dumps(data)
        file.write(json_line + '\n')


# 使用示例
# file_path = 'data.jsonl'
# data = {
#     "trial_index": 2,
#     "large_data": "adfdsad",
# }
# add_to_jsonl(file_path, data)


# jsonl
# [
#     {
#         "trial_index": 1,
#         "comment": "hello!",
#     },
#     {
#         "trial_index": 2,
#         "comment": "fuck!",
#     },
# ]


def concatenate_jsonl(file_path):
    # TODO
    """拼接 comment_history 需要用到，将 jsonl 组织成一整块用于输入 llm"""
    # 使用示例
    # file_path = 'data.jsonl'
    # concatenated_data = concatenate_jsonl(file_path)

    # 效果
    # jsonl
    # [
    #     {
    #         "trial_index": 1,
    #         "comment": "hello!",
    #     },
    #     {
    #         "trial_index": 2,
    #         "comment": "fuck!",
    #     },
    # ]
    # file_path = "./test.json"
    # print(concatenate_jsonl(file_path=file_path))
    # trial_index: 1:
    # comment: hello!
    # trial_index: 2:
    # comment: fuck!
    final_concatenated_data = []

    # Read each line in JSONL file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        for entry in data:
            # Extract keys dynamically
            entry_parts = []
            for key, value in entry.items():
                entry_parts.append(f"{key}: {value}")

            # Join entry parts and add to the final output
            final_concatenated_data.append(":\n".join(entry_parts))

    # Concatenate all data with newlines
    concatenated_output = "\n".join(final_concatenated_data)

    return concatenated_output
