#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-05 21:28:03
@File: src/utils/jsonl.py
@IDE: vscode
@Description:
    封装一些 jsonl 存储与拼接读取的 useful 函数
"""


import json


def add_to_jsonl(file_path, data):
    with open(file_path, 'a') as file:
        json_line = json.dumps(data)
        file.write(json_line + '\n')


def concatenate_jsonl(file_path):
    # TODO
    """拼接 comment_history 需要用到，将 jsonl 组织成一整块用于输入 llm"""
    final_concatenated_data = []

    # Read each line in JSONL file
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as JSON
            data_entry = json.loads(line.strip())

            # Extract `trial_index` and `data` dynamically
            trial_index = list(data_entry.values())[0]
            data_value = list(data_entry.values())[1]

            # Add trial index header and data to final output
            final_concatenated_data.append(f"trial {trial_index}")
            final_concatenated_data.append(data_value)

    # Concatenate all data with newlines
    concatenated_output = "\n".join(final_concatenated_data)

    return concatenated_output
