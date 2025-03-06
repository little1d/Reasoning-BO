#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-04 10:03:29
@File: src/bo/models.py
@IDE: vscode
@Description:
    BO model, Reasoning model
"""

from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
from ax.models.torch.botorch_modular.utils import ModelConfig
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP

import numpy as np
import json
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render
from ax import Trial
import json
from src.config import Config
from typing import Dict
import os

from src.prompts.base import PromptManager
from src.llms.deepseek import DeepSeekClient
from src.utils.metric import save_trial_data

from src.utils.jsonl import add_to_jsonl

config = Config()


class BOModel:
    def __init__(self, experiment):
        self.experiment = experiment
        self.model_bridge = None

    def hot_start(self, experiment):
        NUM_SOBOL_TRIALS = 5
        print(f"Running Sobol initialization trials...")
        # 需要在传递参数前定义好 search_space
        sobol = Models.SOBOL(search_space=experiment.search_space)
        for i in range(NUM_SOBOL_TRIALS):
            generator_run = sobol.gen(n=1)
            trial = experiment.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()

    def gen(self, n):
        self.model_bridge = Models.BOTORCH_MODULAR(
            experiment=self.experiment,
            data=self.experiment.fetch_data(),
            surrogate_spec=SurrogateSpec(
                model_configs=[
                    ModelConfig(botorch_model_class=SingleTaskGP),
                ]
            ),
            botorch_acqf_class=qLogNoisyExpectedImprovement,
        )
        print(f"Running BO trial ...")

        return self.model_bridge.gen(n=n)

    def easy_render_hartmann6(self):
        # 只适用于 gen(n=1)和 notbook，因为 objective_means 只有那个有
        objective_means = np.array(
            [
                [
                    trial.objective_mean
                    for trial in self.experiment.trials.values()
                ]
            ]
        )
        best_objective_plot = optimization_trace_single_method(
            y=np.minimum.accumulate(objective_means, axis=1),
            optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
        )
        render(best_objective_plot)


class DSReasoner:
    def __init__(self, exp_config_path: str, result_dir: str):
        """两个输入参数最好写绝对路径"""
        # ---------------------------------- Experiment Config ----------------------------------
        # print(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
        self.exp_config = self._load_config(exp_config_path)
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        if not self.result_dir.endswith(('/', '\\')):
            self.result_dir = self.result_dir + "/"
        # trial data contains arms and metric, _save_trial_data function takes dir param
        # and create {metric}_.csv automatically
        self.trial_data_dir = self.result_dir + "trial_data/"
        self.messages_file_path = self.result_dir + "messages.jsonl"
        self.experiment_analysis_file_path = (
            self.result_dir + "experiment_analysis.json"
        )
        # ---------------------------------- Object instance----------------------------------
        self.client = DeepSeekClient()
        self.prompt_manager = PromptManager()
        # ---------------------------------- Atributes ----------------------------------
        self.experiment_analysis = {}
        self.overview = ""
        self.summary = ""
        self.conclusion = ""

    def _load_config(self, path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_comment(self, trial_index: int) -> None:
        # 初始化没有comment，comment 是 str，需要转换成 json
        """保存返回的 comment（DSReasoner 的 Assitant 信息） 到 comment history 中，jsonl 格式"""
        new_comment = self.client.messages[-1]['content']  # json
        data = {"trial_index": trial_index, "comment": new_comment}  # dict
        add_to_jsonl(self.comment_history_file_path, data)

    def _save_messages(self):
        self.client.save_messages(self.messages_file_path)

    def _extract_candidates_from_comment(
        self,
    ):
        """返回置信度最高的 5 个 candidates"""

    def generate_overview(self) -> str:
        try:
            print("Start generating overview...")
            formatted_prompt = self.prompt_manager.format(
                "overview_generate", **self.exp_config
            )
            # print(f"Formatted prompt: {formatted_prompt}")

            content, _ = self.client.generate(user_prompt=formatted_prompt)

            self.overview = content

            print(
                f"Overview has been generated! and the content is as follows\n {content}"
            )
            return content

        except Exception as e:
            print(f"Error generating overview: {e}")
            return ""

    def initial_sampling(self) -> str:
        """在 initial_sampling，没有保存 messages 和 trial_data。
        # TODO 加错误处理
        有个问题：没有overview 也能sample..."""
        try:
            print("Start initial sampling...")
            meta_dict = {
                **self.exp_config,
                "overview": self.overview,
            }
            formatted_prompt = self.prompt_manager.format(
                "initial_sampling", **meta_dict
            )
            content, _ = self.client.generate(user_prompt=formatted_prompt)
            print(
                f"Initial sampling process has done! and the content is as follows\n {content}\n\n"
            )
            self._save_messages()
            return content

        except Exception as e:
            print(f"Error happended while initial sampling: {e}")
            return ""

    def optimization_first_round(
        self,
    ):
        """take in -> (rag) -> generate(and save) -> comment -> return candidates(extract_comment_from_candidates)"""

    def optimization_loop(self, experiment, trial: Trial) -> str:
        """根据上一轮的trial data(arms, metrics), comment history, 生成下一轮的 comment"""

        save_trial_data(
            experiment=experiment, trial=trial, save_dir=self.trial_data_dir
        )
        self._save_comment(iteration=trial.index)
        self._save_messages()
        pass

    def _generate_summary(self):
        """返回 json 格式"""
        print(f"generating summary...")
        pass

    def _generate_conclusion(self):
        """返回 json 格式"""
        print(f"generating conclusion...")
        pass

    def generate_experiment_analysis(self):
        """overview + summary + conclusion, 从 self 里面拿，反正不是很多"""
        file_path = self.result_dir + "experiment_analysis.json"

        data_dict = {
            "overview": self._generate_summary(),
            "summary": self._generate_conclusion(),
            "conclusion": self.conclusion,
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=4)


# class O1Reasoner:
#     pass
