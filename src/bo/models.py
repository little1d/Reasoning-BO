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
import json
from src.config import Config
from typing import Dict

from src.prompts.base import PromptManager
from src.llms.deepseek import DeepSeekClient


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
    def __init__(self, exp_config_path: str):
        # 最好写绝对路径吧
        self.exp_config = self._load_config(exp_config_path)
        self.client = DeepSeekClient()
        self.prompt_manager = PromptManager()
        self.overview = ""
        self.summary = ""
        self.experiment_conclusion = ""

    def _load_config(self, path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_comment(self, comment_history_file_path) -> None:
        """保存返回的 comment（DSReasoner 的 Assitant 信息） 到 comment history 中，jsonl 格式"""
        pass

    def _save_trial_data(self):
        pass

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
        # TODO
        """没有overview 也能sample..."""
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
                f"initial sampling process has done! and the content is as follows\n {content}"
            )
            return content

        except Exception as e:
            print(f"Error happended while initial sampling: {e}")
            return ""

    def optimization_loop(self) -> str:
        """根据上一轮的trial data(arms, metrics), comment history, 生成下一轮的 comment"""
        pass


# class O1Reasoner:
#     pass
