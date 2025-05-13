import json
from ax import Trial, Arm, GeneratorRun, Experiment
import json
from typing import Dict
import os
import re
import glob
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from src.prompts.base import PromptManager
from src.utils.metric import save_trial_data

from src.utils.jsonl import add_to_jsonl, concatenate_jsonl
from src.bo.models import BOModel
from src.config import Config

config = Config()

system_message = "You are an expert at extracting json from text."


class BaseReasoner:
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
        self.insight_history_file_path = (
            self.result_dir + "insight_history.jsonl"
        )
        self.experiment_analysis_file_path = (
            self.result_dir + "experiment_analysis.jsonl"
        )
        # ---------------------------------- Object instance----------------------------------
        self.prompt_manager = PromptManager()
        # ---------------------------------- Atributes ----------------------------------
        self.experiment_analysis = {}
        self.overview = ""
        self.summary = ""
        self.report = ""
        self.keywords = ""
        # ---------------------------------- chatagent init ----------------------------------
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            api_key=config.QWQ_API_KEY,
            url=config.QWQ_API_BASE,
            model_type=config.NOTES_AGENT,
            model_config_dict={"temperature": 0, "max_tokens": 2048},
        )
        # formatter
        self.chat_agent = ChatAgent(
            model=self.model,
            system_message=system_message,
        )

    def _get_system_prompt_template(self, raw_insight: str) -> str:
        """Helper method to generate the system prompt template with raw_insight"""
        return f"""The following text contains a JSON object that may be wrapped in markdown or other formatting. 
        Please extract just the JSON object and ensure it is properly formatted. 
        Return ONLY the JSON object with no additional text or explanation.

        Text to clean:
        {raw_insight}"""

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

    def _save_insight(self, trial_index: int) -> None:
        # 初始化没有insight，insight 是 str，需要转换成 json
        """保存返回的 insight（DSReasoner 的 Assitant 信息） 到 insight history 中，jsonl 格式"""
        print(f"Start saving the insight data for this round of trials\n")
        new_insight = self.client.messages[-1]['content']  # json
        data = {"trial_index": trial_index, "insight": new_insight}  # dict
        add_to_jsonl(self.insight_history_file_path, data)
        print("Done!\n")

    def _save_messages(self):
        self.client.save_messages(self.messages_file_path)

    def _extract_keywords_from_insight(self, insight: str):
        """从insight中提取用于检索的关键词

        Args:
            insight: 包含keywords字段的JSON字符串

        Returns:
            提取到的关键词字符串，如果提取失败则返回空字符串
        """
        try:
            insight = insight.strip()
            # 移除可能的JSON代码块标记
            insight = re.sub(
                r'^```json\s*|\s*```$', '', insight, flags=re.MULTILINE
            )

            # 解析JSON
            insight_data = json.loads(insight)

            # 检查keywords字段是否存在且是字符串
            if isinstance(insight_data, dict) and "keywords" in insight_data:
                keywords = insight_data["keywords"]
                if isinstance(keywords, str):
                    return keywords.strip()
                elif isinstance(keywords, (list, tuple)):
                    return " ".join(str(k) for k in keywords).strip()

            print(f"Warning: No valid 'keywords' field found in insight")
            return ""

        except json.JSONDecodeError:
            print(f"Warning: Failed to parse insight as JSON")
            return ""
        except Exception as e:
            print(f"Warning: Unexpected error extracting keywords - {str(e)}")
            return ""

    def get_keywords(self):
        return self.keywords

    def _extract_candidates_from_insight(self, insight, n: int = 3):
        # BUG
        # 对于复杂任务，insight 给出的 candidates 很可能有错误，这个函数很可能抽取三个 candidates都不是有效的 arms，导致算法中断
        """输入 insight(json)，返回置信度最高的 n 个 candidates"""
        print("Start extracting candidates array from insight...")
        insight = insight.strip()
        insight = re.sub(
            r'^```json\s*|\s*```$', '', insight, flags=re.MULTILINE
        )

        CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}
        # json to dict
        insight = json.loads(insight)
        if not isinstance(insight, dict) or "hypotheses" not in insight:
            raise ValueError("Invalid JSON format: missing 'hypotheses' key")

        sorted_hypotheses = sorted(
            insight["hypotheses"],
            key=lambda x: CONFIDENCE_ORDER.get(x["confidence"].lower(), 3),
        )

        candidates = []
        for hyp in sorted_hypotheses:
            if "parameter_sets" not in hyp or not isinstance(
                hyp["parameter_sets"], list
            ):
                continue

            for point in hyp["parameter_sets"]:
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
        """保存实验数据，包括 insight, messages 和 trial_data"""
        print(
            "Start saving the experiment data, including insight, messages and trial data...\n"
        )
        self._save_insight(trial_index=trial.index)
        self._save_messages()
        save_trial_data(
            experiment=experiment, trial=trial, save_dir=self.trial_data_dir
        )
        print("Done!\n")

    def generate_overview(self) -> str:
        try:
            print("Start generating overview...")
            formatted_prompt = self.prompt_manager.format(
                "generate_overview", **self.exp_config
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
            raw_insight, _ = self.client.generate(user_prompt=formatted_prompt)
            chat_agent_prompts = self._get_system_prompt_template(raw_insight)
            insight = self.chat_agent.step(
                chat_agent_prompts,
            ).msg.content
            print(
                f"Initial sampling process has done! and the insight is as follows\n {insight}\n\n"
            )

            return insight

        except Exception as e:
            print(f"Error happended while initial sampling: {e}")
            return ""

    def optimization_first_round(self, insight):
        # 第一轮并没有 Trial，所以 optimization 中不保存任何数据，单独处理！在外面运行完实验后动态保存
        candidates = self._extract_candidates_from_insight(insight)
        self.keywords = self._extract_keywords_from_insight(insight)
        return candidates

    def optimization_loop(
        self,
        experiment: Experiment,
        trial: Trial,
        bo_model: BOModel,
        retrieval_context: str = None,
        n: int = 7,
    ) -> str:
        """take in -> (rag) -> generate(and save) -> insight -> return candidates(extract_insight_from_candidates)"""
        """根据上一轮的trial data(arms, metrics), insight history, 生成下一轮的 insight，并返回 candidates_array"""

        """Perform optimization loop to generate insights and candidates for next trial.
        Args:
            experiment: Current experiment object containing trial data
            trial: Current trial object
            bo_model: Bayesian Optimization model used for generating recommendations
            retrieval_context: Optional context from retrieval system, if None, represent no retrieve
            n: Number of candidates for bo_model to generate
            
        Returns:
            str: Generated 'insight' text containing analysis and recommendations
            
        The function:
            1. Gets BO model recommendations
            2. Loads trial data and insight history 
            3. Formats prompt with experiment config, trial data, insights etc.
            4. Generates new insight using client
            5. Extracts keywords and candidates from insight
        """
        # 获取BO模型推荐的点
        generator_run_by_bo = bo_model.gen(n=n)
        bo_candidates = [arm.parameters for arm in generator_run_by_bo.arms]

        # 加载 trial data， dir（self.trial_data_dir） 下面包含所有的 metrics.csv 文件
        trial_data = self._load_trial_data()
        # 加载 insight history   self.insight_history_file_path  是一个 jsonl 文件，可以通过concatenate_jsonl函数拼接 insight_history

        with open(self.insight_history_file_path, 'r', encoding='utf-8') as f:
            insight_history = []
            for line_number, line in enumerate(f, 1):
                stripped_line = line.strip()
                # 跳过空行
                if not stripped_line:
                    continue
                try:
                    insight_history.append(json.loads(stripped_line))
                except json.JSONDecodeError as e:
                    print(
                        f"第{line_number}行解析失败，内容：{stripped_line[:50]}...，错误类型：{e.msg}"
                    )

        # 将解析后的JSON对象列表传递给concatenate_jsonl
        insight_history = concatenate_jsonl(insight_history)
        # 利用 prompt_template 生成 prompt。并使用dsreasoner.generate 生成 insight
        condidates_array = []
        try:
            print(f"Start Optimization iteration {trial.index}...")
            # 把trial data, insight history 拼接到 meta_dict 中
            meta_dict = {
                **self.exp_config,
                "iteration": trial.index,
                "trial_data": trial_data,
                "insight_history": insight_history,
                "bo_recommendations": bo_candidates,
                "retrieved_context": retrieval_context,
            }
            # 利用 prompt template "optimization loop" 生成 formatted_prompt
            formatted_prompt = self.prompt_manager.format(
                "optimization_loop", **meta_dict
            )
            raw_insight, _ = self.client.generate(user_prompt=formatted_prompt)
            chat_agent_prompts = self._get_system_prompt_template(raw_insight)
            insight = self.chat_agent.step(
                chat_agent_prompts,
            ).msg.content

            print(
                f"Optimization loop iteration {trial.index} has done! and the insight is as follows\n {insight}\n\n"
            )
            self.keywords = self._extract_keywords_from_insight(insight)
            condidates_array = self._extract_candidates_from_insight(insight)

        except Exception as e:
            print(
                f"Error happended while optimization iteration {trial.index}: {e}"
            )
            return ""

        # 保存数据
        self._save_experiment_data(experiment=experiment, trial=trial)

        # 从insight中抽象candidates_array 并 return
        return condidates_array

    def _generate_summary(self, trial_data, insight_history):
        """返回 json 格式，其实 markdown 更贴切"""
        print(f"Start generating summary...\n")
        meta_dict = {
            **self.exp_config,
            "iteration": len(insight_history),
            "trial_data": trial_data,
            "insight_history": insight_history,
        }
        formatted_prompt = self.prompt_manager.format(
            "generate_summary", **meta_dict
        )
        insight, _ = self.client.generate(user_prompt=formatted_prompt)
        print(
            f"Experiment summary has been generated! and the insight is as follows\n {insight}\n\n"
        )
        self.summary = insight
        self._save_messages()
        return insight

    def _generate_report(self, trial_data, insight_history):
        """返回 json 格式"""
        print(f"Start generating report...\n")
        meta_dict = {
            **self.exp_config,
            "iteration": len(insight_history),
            "trial_data": trial_data,
            "insight_history": insight_history,
        }
        formatted_prompt = self.prompt_manager.format(
            "generate_report", **meta_dict
        )
        insight, _ = self.client.generate(user_prompt=formatted_prompt)
        print(
            f"Experiment summary has been generated! and the insight is as follows\n {insight}\n\n"
        )
        self.report = insight
        # 作为调试，看看user_input
        self._save_messages()
        return insight

    def generate_experiment_analysis(
        self,
    ):
        """overview + summary + report, 从 self 里面拿，反正不是很多"""
        print(
            "Start generating experiment analysis..., conluding overview, experiment summary and report. \n"
        )
        file_path = self.result_dir + "experiment_analysis.json"
        trial_data = self._load_trial_data()
        with open(self.insight_history_file_path, 'r', encoding='utf-8') as f:
            # 逐行读取并解析JSONL文件
            insight_history = [json.loads(line) for line in f]

        # 将解析后的JSON对象列表传递给concatenate_jsonl
        insight_history = concatenate_jsonl(insight_history)
        analysis = {
            "overview": self.overview,
            "summary": self._generate_summary(trial_data, insight_history),
            "report": self._generate_report(trial_data, insight_history),
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=4)
        print("Done!\n")
