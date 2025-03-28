import json
from ax import Trial, Arm, GeneratorRun
import json
from typing import Dict
import os
import re
import glob

from src.prompts.base import PromptManager
from src.llms.qwq import QWQClient
from src.utils.metric import save_trial_data

from src.utils.jsonl import add_to_jsonl, concatenate_jsonl
from src.bo.models import BOModel


class QWQReasoner:
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
        self.comment_history_file_path = (
            self.result_dir + "comment_history.jsonl"
        )
        self.experiment_analysis_file_path = (
            self.result_dir + "experiment_analysis.jsonl"
        )
        # ---------------------------------- Object instance----------------------------------
        self.client = QWQClient()
        self.prompt_manager = PromptManager()
        # ---------------------------------- Atributes ----------------------------------
        self.experiment_analysis = {}
        self.overview = ""
        self.summary = ""
        self.conclusion = ""

    def _load_config(self, path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_trial_data(self):
        """Load trial data from multiple CSV files as a combined string"""
        print(
            "Start Loading trial data from multiple CSV files as a combined string"
        )
        csv_files = glob.glob(os.path.join((self.trial_data_dir), "*.csv"))
        combined_data = []
        for file_path in csv_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                combined_data.append(file.read())
        print("Done!\n")
        return "\n".join(combined_data)

    def _save_comment(self, trial_index: int) -> None:
        # 初始化没有comment，comment 是 str，需要转换成 json
        """保存返回的 comment（DSReasoner 的 Assitant 信息） 到 comment history 中，jsonl 格式"""
        print(f"Start saving the comment data for this round of trials\n")
        new_comment = self.client.messages[-1]['content']  # json
        data = {"trial_index": trial_index, "comment": new_comment}  # dict
        add_to_jsonl(self.comment_history_file_path, data)
        print("Done!\n")

    def _save_messages(self):
        self.client.save_messages(self.messages_file_path)

    def _extract_candidates_from_comment(self, comment, n: int = 5):
        """输入 comment(json)，返回置信度最高的 n 个 candidates"""
        print("Start extracting candidates array from comment...")
        comment = comment.strip()
        comment = re.sub(
            r'^```json\s*|\s*```$', '', comment, flags=re.MULTILINE
        )

        CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}
        # json to dict
        comment = json.loads(comment)
        if not isinstance(comment, dict) or "hypotheses" not in comment:
            raise ValueError("Invalid JSON format: missing 'hypotheses' key")

        sorted_hypotheses = sorted(
            comment["hypotheses"],
            key=lambda x: CONFIDENCE_ORDER.get(x["confidence"].lower(), 3),
        )

        candidates = []
        for hyp in sorted_hypotheses:
            if "points" not in hyp or not isinstance(hyp["points"], list):
                continue

            for point in hyp["points"]:
                if not isinstance(point, dict):
                    continue
                candidates.append(point)
                if len(candidates) == n:
                    print(
                        f"Done! We have collected {n} candidates, The candidates points is as follows{candidates}"
                    )
                    return candidates
        # 如果可用点不足 n 个，全部返回
        print(
            f"Done! We have collected candidates less than {n}. The candidates points is as follows{candidates}"
        )
        return candidates

    def run_bo_experiment(self, experiment, candidates_array):
        """运行一轮实验，包括创建 trial、运行和标记完成"""
        print("Start running bo experiment...\n")
        candidates = [Arm(parameters=params) for params in candidates_array]
        filtered_generator_run = GeneratorRun(arms=candidates)
        trial = experiment.new_batch_trial(
            generator_run=filtered_generator_run
        )
        trial.run()
        trial.mark_completed()
        print("Done!\n")
        return trial

    def _save_experiment_data(self, experiment, trial: Trial) -> None:
        """保存实验数据，包括 comment, messages 和 trial_data"""
        print(
            "Start saving the experiment data, including comment, messages and trial data...\n"
        )
        self._save_comment(trial_index=trial.index)
        self._save_messages()
        save_trial_data(
            experiment=experiment, trial=trial, save_dir=self.trial_data_dir
        )
        print("Done!\n")

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
                f"Overview has been generated! and the content is as follows\n {content}\n"
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
                f"Initial sampling process has done! and the comment is as follows\n {content}\n\n"
            )
            return content

        except Exception as e:
            print(f"Error happended while initial sampling: {e}")
            return ""

    def optimization_first_round(self, comment):
        # 第一轮并没有 Trial，所以 optimization 中不保存任何数据，单独处理！在外面运行完实验后动态保存
        candidates = self._extract_candidates_from_comment(comment)
        return candidates

    def optimization_loop(
        self, experiment, trial: Trial, bo_model: BOModel, n: int = 10
    ) -> str:
        """take in -> (rag) -> generate(and save) -> comment -> return candidates(extract_comment_from_candidates)"""
        """根据上一轮的trial data(arms, metrics), comment history, 生成下一轮的 comment，并返回 candidates_array"""
        # 获取BO模型推荐的点
        generator_run_by_bo = bo_model.gen(n=n)
        bo_candidates = [arm.parameters for arm in generator_run_by_bo.arms]

        # 加载 trial data， dir（self.trial_data_dir） 下面包含所有的 metrics.csv 文件
        trial_data = self._load_trial_data()
        # 加载 comment history   self.comment_history_file_path  是一个 jsonl 文件，可以通过concatenate_jsonl函数拼接 comment_history

        with open(self.comment_history_file_path, 'r', encoding='utf-8') as f:
            comment_history = []
            for line_number, line in enumerate(f, 1):
                stripped_line = line.strip()
                # 跳过空行
                if not stripped_line:
                    continue
                try:
                    comment_history.append(json.loads(stripped_line))
                except json.JSONDecodeError as e:
                    print(
                        f"第{line_number}行解析失败，内容：{stripped_line[:50]}...，错误类型：{e.msg}"
                    )

        # 将解析后的JSON对象列表传递给concatenate_jsonl
        comment_history = concatenate_jsonl(comment_history)
        # 利用 prompt_template 生成 prompt。并使用dsreasoner.generate 生成 comment
        condidates_array = []
        try:
            print(f"Start Optimization iteration {trial.index}...")
            # 把trial data, comment history 拼接到 meta_dict 中
            meta_dict = {
                **self.exp_config,
                "iteration": trial.index,
                "trial_data": trial_data,
                "comment_history": comment_history,
                "bo_recommendations": bo_candidates,
            }
            # 利用 prompt template "optimization loop" 生成 formatted_prompt
            formatted_prompt = self.prompt_manager.format(
                "optimization_loop", **meta_dict
            )
            comment, _ = self.client.generate(user_prompt=formatted_prompt)
            print(
                f"Optimization loop iteration {trial.index} has done! and the comment is as follows\n {comment}\n\n"
            )
            condidates_array = self._extract_candidates_from_comment(comment)

        except Exception as e:
            print(
                f"Error happended while optimization iteration {trial.index}: {e}"
            )
            return ""

        # 保存数据
        self._save_experiment_data(experiment=experiment, trial=trial)

        # 从comment中抽象candidates_array 并 return
        return condidates_array

    def _generate_summary(self, trial_data, comment_history):
        """返回 json 格式，其实 markdown 更贴切"""
        print(f"Start generating summary...\n")
        meta_dict = {
            **self.exp_config,
            "iteration": len(comment_history),
            "trial_data": trial_data,
            "comment_history": comment_history,
        }
        formatted_prompt = self.prompt_manager.format(
            "generate_summary", **meta_dict
        )
        comment, _ = self.client.generate(user_prompt=formatted_prompt)
        print(
            f"Experiment summary has been generated! and the comment is as follows\n {comment}\n\n"
        )
        self.summary = comment
        self._save_messages()
        return comment

    def _generate_conclusion(self, trial_data, comment_history):
        """返回 json 格式"""
        print(f"Start generating conclusion...\n")
        meta_dict = {
            **self.exp_config,
            "iteration": len(comment_history),
            "trial_data": trial_data,
            "comment_history": comment_history,
        }
        formatted_prompt = self.prompt_manager.format(
            "generate_conclusion", **meta_dict
        )
        comment, _ = self.client.generate(user_prompt=formatted_prompt)
        print(
            f"Experiment summary has been generated! and the comment is as follows\n {comment}\n\n"
        )
        self.conclusion = comment
        # 作为调试，看看user_input
        self._save_messages()
        return comment

    def generate_experiment_analysis(
        self,
    ):
        """overview + summary + conclusion, 从 self 里面拿，反正不是很多"""
        print(
            "Start generating experiment analysis..., conluding overview, experiment summary and conclusion. \n"
        )
        file_path = self.result_dir + "experiment_analysis.json"
        trial_data = self._load_trial_data()
        with open(self.comment_history_file_path, 'r', encoding='utf-8') as f:
            # 逐行读取并解析JSONL文件
            comment_history = [json.loads(line) for line in f]

        # 将解析后的JSON对象列表传递给concatenate_jsonl
        comment_history = concatenate_jsonl(comment_history)
        analysis = {
            "overview": self.overview,
            "summary": self._generate_summary(trial_data, comment_history),
            "conclusion": self._generate_conclusion(
                trial_data, comment_history
            ),
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=4)
        print("Done!\n")
