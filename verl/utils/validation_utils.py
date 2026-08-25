# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
工具函数用于处理验证相关的操作，比如保存验证结果到JSON文件。
"""

import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_validation_results_to_json(data_sources: List[str],
                                    sample_inputs: List[str],
                                    infos_dict: Dict[str, List[Any]],
                                    json_path: str = "validation_results.json") -> str:
    """将验证结果保存到JSON文件中，按数据源和提示分组。
    
    Args:
        data_sources: 每个样本的数据源标识符列表
        sample_inputs: 输入提示列表
        infos_dict: 变量名到每个样本值列表的映射
        json_path: 保存JSON文件的路径
        
    Returns:
        str: 实际保存的文件路径（包含时间戳）
    """
    # 创建用于JSON导出的数据结构
    json_results = defaultdict(lambda: defaultdict(list))
    
    for sample_idx, data_source in enumerate(data_sources):
        if sample_idx >= len(sample_inputs):
            continue
            
        prompt = sample_inputs[sample_idx]
        
        # 收集用于JSON导出的数据
        sample_data = {}
        sample_data["prompt"] = prompt
        sample_data["data_source"] = data_source
        
        for var_name, var_vals in infos_dict.items():
            if sample_idx < len(var_vals):
                val = _jsonable(var_vals[sample_idx])
                # 存储所有感兴趣的字段到JSON
                if var_name == "reward":
                    sample_data["score"] = val
                elif var_name == "pred":
                    sample_data["prediction"] = val
                elif var_name == "genrm_pred":
                    sample_data["genrm_prediction"] = val
                elif var_name == "acc":
                    sample_data["is_correct"] = bool(val >= 0.5)
                elif var_name == "ground_truth":
                    sample_data["ground_truth"] = val
                elif var_name == "response":
                    sample_data["response"] = val
                elif var_name == "genrm_score":
                    sample_data["genrm_score"] = val
                elif var_name in (
                    "acc_t1", "acc_t2", "acc_final", "pred_t1", "pred_t2",
                    "revised", "final_turn", "ground_truth", "genrm_probs",
                    "data_source", "acc_generic", "r_critique", "pair_cell",
                    "verify_text", "generic_response",
                    "p_regen", "regen_weight", "r_self", "delta_self", "r_feedback",
                ):
                    sample_data[var_name] = val
        
        # 将样本数据添加到对应提示的列表中
        if sample_data:
            json_results[data_source][prompt].append(sample_data)

    # 添加时间戳到文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{os.path.splitext(json_path)[0]}_{timestamp}.json"
    
    # 将嵌套的defaultdict转换为普通dict
    json_dict = {}
    for data_source, prompt_dict in json_results.items():
        json_dict[data_source] = {}
        for prompt, samples in prompt_dict.items():
            json_dict[data_source][prompt] = samples
    
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # 保存到文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=2)
    
    print(f"验证结果已保存到: {filename}")
    
    return filename 