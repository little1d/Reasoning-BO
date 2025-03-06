#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-04 11:21:44
@File: src/utils/metric.py
@IDE: vscode
@Description:
    从 ax 到结果存储的转换函数，还包括一些 io 函数
"""
import numpy as np
from ax import Experiment, Trial
import csv
import os


def save_trial_data(
    experiment: Experiment,
    trial: Trial,
    save_dir: str,
    filename: str,
) -> None:
    """进行一轮 Trial/BatchTrial 后，保存 arms 和 metrics到本地，作为参考在下一轮提供给 llm

    Parameters
    ----------
    experiment : Experiment
        当前实验实例
    trial : Trial/BatchTrial
        当前 trial 实例，可为 Trial 和 BatchTrial
    save_dir : str
        json 和 csv 文件保存目录
    filename : str
        json 和 csv 文件名
    """
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    data_df = experiment.fetch_trials_data(trial_indices=[trial.index]).df

    # 构建数据记录
    # arms需要从 trial object 拿，metric 要从 experiment 对象拿。先全部记录成 record，再存储
    # 没办法，我也不想写💩山的
    record = {
        "trial_index": trial.index,
        "arms_parameters": [
            {
                "arm_name": arm.name,
                **{
                    key: round(value, 3)
                    for key, value in arm.parameters.items()
                },
            }
            for arm in trial.arms
        ],
        "metric_data": [
            {
                "arm_name": row["arm_name"],
                "metric_name": row["metric_name"],
                "mean": row["mean"],  # 保留原始精度
            }
            for _, row in data_df.iterrows()
        ],
    }
    # json_file = os.path.join(save_dir, f"{filename}.json")

    # 保存JSON，但其实没啥必要，可以调试，看看这一轮实验
    # with open(json_file, 'a+') as f:
    #     json.dump(record, f, indent=2)
    #     f.write('\n')

    # 生成结构化CSV表格
    # 获取参数列名（排除arm_name）
    param_columns = [
        k for k in record["arms_parameters"][0] if k != "arm_name"
    ]

    # 按metric分组生成表格
    metrics = {m["metric_name"] for m in record["metric_data"]}

    for metric in metrics:
        csv_filename = os.path.join(save_dir, f"{filename}_{metric}.csv")
        rows = []

        # 构建表格行数据
        for arm in record["arms_parameters"]:
            # 获取该arm在当前 metric 的mean值
            mean_value = next(
                (
                    m["mean"]
                    for m in record["metric_data"]
                    if m["arm_name"] == arm["arm_name"]
                    and m["metric_name"] == metric
                ),
                None,
            )

            if mean_value is not None:
                row = {
                    "trial_index": arm["arm_name"],
                    **{col: arm[col] for col in param_columns},
                    f"{metric}_mean_value": round(mean_value, 3),
                }
                rows.append(row)

        # 按arm_name排序
        rows.sort(key=lambda x: [int(n) for n in x["trial_index"].split('_')])

        # 写入CSV文件，增量存储
        with open(csv_filename, 'a+', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=["arm_name"] + param_columns + ["mean"]
            )
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)


def extract_metric(exp: Experiment, metric_name: str) -> np.ndarray:
    """
    提取实验对象中指定metric的均值序列，用于所有实验结束后，结果的展示。其实在 save_trial_data 函数中，会把
    所有的 metrics 增量存储到 xxx_metric.csc 中，但那是以 arm 为最小单位的。对于 BatchTrial 来说，一个 Batch
    里不同 arm 的 metric 值可能相差很大。针对最后结果绘图而言，取均值是很合理的，可以反映上一轮 sample 的 candidates 整体水平，
    随着实验的进行，mean 理论上是逐渐升高的。
    参数：
        exp: 实验对象
        metric_name: 需要提取的指标名称（如'hartman6'），在 objective/constriant 中定义的"name"一样
    返回：
        np.ndarray: 按 trial_index 顺序排列的指标均值数组
        对于BatchTrial自动取该批次的均值
    """
    # 获取原始数据并过滤目标指标
    df = exp.fetch_data().df
    metric_df = df[df['metric_name'] == metric_name]

    # 按trial分组处理均值
    results = []
    for trial_idx, group in metric_df.groupby('trial_index'):
        # 判断是否为BatchTrial（依据每组arm数量）
        if len(group) > 1:
            batch_mean = group['mean'].mean()
            results.append((trial_idx, batch_mean))
        else:
            results.append((trial_idx, group['mean'].iloc[0]))

    # 按trial顺序排列并转换为数组
    sorted_results = sorted(results, key=lambda x: x[0])
    return np.array([val for _, val in sorted_results])
