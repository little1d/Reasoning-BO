#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-02-27 16:06:39
@File: src/app/config.py
@IDE: vscode
@Description:
    Store environment variables
"""

import os
from dotenv import load_dotenv

# 系统环境变量优先级 > .env 配置的值
load_dotenv('.env', override=False)


class Config:
    def __init__(self):
        self.OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
        self.NEO4J_URL = "bolt://10.140.52.87:7687"
        self.NEO4J_USERNAME = "neo4j"
        self.NEO4J_PASSWORD = "123456789"
        self.MILVUS_HOST = "10.140.52.87"
        self.MILVUS_PORT = "19530"
