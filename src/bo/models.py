#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-04 10:03:29
@File: src/bo/models.py
@IDE: vscode
@Description:
    BO model
"""

from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
from ax.models.torch.botorch_modular.utils import ModelConfig
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP

import numpy as np
import json
from datetime import datetime
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render


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


# 保存和加载实验的逻辑应该在外面，因为可能要重写 metrics，不适合与模型耦合


import json
import re
from datetime import datetime
from openai import OpenAI
import os


class DSReasoner:
    def __init__(self, api_key, workdir):
        self.client = OpenAI(
            api_key=api_key, base_url="https://api.deepseek.com"
        )
        self.workdir = workdir
        self.comment_history = []
        self.prompt_template = (
            "<think>Based on current candidates and accumulated comments, "
            "make a recommendation for the best points.</think>\n"
            "Previous comments: {}\n"
            "Hypothesis: {}\n"
            "Candidates: {}"
        )

    def _parse_response(self, response):
        thought = re.findall(r'<think>(.*?)', response, re.DOTALL)
        conclusion = re.sub(
            r'<think>.*?', '', response, flags=re.DOTALL
        ).strip()
        return thought[0] if thought else "", conclusion

    def recommend(self, candidates, iteration, trial_data):
        try:
            # 准备统计指标
            stats = {
                "n_trials": len(trial_data),
                "best_score": (
                    max(t.get("score", 0) for t in trial_data)
                    if trial_data
                    else 0
                ),
            }

            # 构建结构化提示词
            prompt = self.prompt_template.format(
                comment_history="\n".join(self.comment_history[-3:]),
                n_candidates=len(candidates),
                candidates=json.dumps(candidates, indent=2),
                **stats,
            )

            # 调用DeepSeek-R1模型
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw_response = response.choices[0].message.content

            # 解析响应
            thought_process, conclusion = self._parse_response(raw_response)
            response_json = json.loads(conclusion)

            # 筛选候选
            filtered_ids = response_json["recommendations"]
            filtered_candidates = [candidates[i] for i in filtered_ids]

            return filtered_candidates, raw_response

        except Exception as e:
            print(f"API Error: {str(e)}")
            return candidates[:1], f"Error: {str(e)}"

    def update(self, iteration, raw_response):
        thought, conclusion = self._parse_response(raw_response)
        self.comment_history.append(
            {
                "iteration": iteration,
                "thought": thought,
                "conclusion": conclusion,
            }
        )

        data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "raw_response": raw_response,
            "parsed": {"thought": thought, "conclusion": conclusion},
        }

        try:
            os.makedirs(self.workdir, exist_ok=True)
            path = os.path.join(self.workdir, f"iter_{iteration}.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Save failed: {str(e)}")
