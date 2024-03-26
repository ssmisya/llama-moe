import json

import numpy as np
import torch

LLAMA_DATA_PORTION = {
    "en_cc": 0.67,
    "en_c4": 0.15,
    "github": 0.045,
    "en_wikipedia": 0.045,
    "en_book": 0.045,
    "en_arxiv": 0.025,
    "en_stack": 0.02,
}

SHEAREDLLAMA_DATA_PORTION = {
    "en_cc": 0.361,
    "en_c4": 0.492,
    "github": 0.008,
    "en_wikipedia": 0.031,
    "en_book": 0.091,
    "en_arxiv": 0.007,
    "en_stack": 0.01,
}

CODE_DATA_PORTION = {
    "en_cc": 0.07,
    "the_stack": 0.85,
    "en_stack": 0.08,
}

CODE_DATA_PORTION_ONLY_EVAL = {
    "shell": 0.005778694348161168,
    "cpp": 0.08471646234734545,
    "c-sharp": 0.05679103733642001,
    "d": 4.4668326078897255e-06,
    "go": 0.04699860994091657,
    "java": 0.11918069344999194,
    "javascript": 0.24993826230612898,
    "julia": 0.0028270617590372446,
    "lua": 0.006004539875253492,
    "perl": 0.004935537989317665,
    "php": 0.09690155857042208,
    "r": 0.0006702627312029321,
    "racket": 6.376433661955171e-05,
    "ruby": 0.010352621260245713,
    "rust": 0.013886281444740753,
    "scala": 0.006728020314970834,
    "swift": 0.007745542071831382,
    "typescript": 0.04191739187878124,
    "python": 0.09455919120600505,
    "en_cc": 0.07,
    "en_stack": 0.08,
}

# CODE_DATA_PORTION_FULL=None

with open(
    "/mnt/petrelfs/songmingyang/code/llama-moe/smoe/data/data_portion/full_language_portion.json",
    "r",
) as f:
    # 加载 JSON 数据
    CODE_DATA_PORTION_FULL = json.load(f)

AVERAGE_SLIMPAJAMA_DATA_PORTION = {
    "en_cc": 1 / 7,
    "en_c4": 1 / 7,
    "github": 1 / 7,
    "en_wikipedia": 1 / 7,
    "en_book": 1 / 7,
    "en_arxiv": 1 / 7,
    "en_stack": 1 / 7,
}

LLAMA2_7B_SLIMPAJAMA_VAL_REF_LOSS = {
    "en_book": 1.925248146057129,
    "en_wikipedia": 1.5899001359939575,
    "en_stack": 1.4974864721298218,
    "github": 0.6984495520591736,
    "en_c4": 2.074881076812744,
    "en_cc": 1.6916865110397339,
    "en_arxiv": 1.2408167123794556,
}


"""
llama2-7B

hellaswag: 2.664067268371582
mmlu: 2.3618555068969727
arc_challenge: 3.6212270259857178
gsm8k: 1.280044436454773
"""


def update_weight_sheared_llama_paper(
    prob_map: dict[str, float], ref_loss: dict[str, float], curr_loss: dict[str, float]
) -> dict[str, float]:
    """
    Args:
        prob_map: dataset name -> prob
        ref_loss: dataset name -> ref loss
        curr_loss: dataset name -> curr loss

    Returns:
        prob_map: updated prob map

    References:
        Dynamic Batch Loading in ShearedLlama (http://arxiv.org/abs/2310.06694)
    """
    task_types = [k for k in prob_map]
    original_weight = np.array([prob_map[k] for k in task_types])
    loss_delta = np.array([max(0, curr_loss[k] - ref_loss[k]) for k in task_types])

    # original method
    alpha = original_weight * np.exp(loss_delta)
    alpha /= alpha.sum()

    # method 2
    # ref_loss_arr = np.array([ref_loss[k] for k in task_types])
    # alpha = original_weight * np.exp(loss_delta / ref_loss_arr)
    # alpha /= alpha.sum()

    # method 3
    # curr_loss_arr = np.array([curr_loss[k] for k in task_types])
    # ref_loss_arr = np.array([ref_loss[k] for k in task_types])
    # loss_delta_arr = curr_loss_arr - ref_loss_arr
    # # loss_delta_arr /= ref_loss_arr
    # lr = 1.0
    # alpha = original_weight + lr * loss_delta_arr

    return {k: v for k, v in zip(task_types, alpha)}


def update_weight_sheared_llama(
    prob_map: dict[str, float], ref_loss: dict[str, float], curr_loss: dict[str, float]
) -> dict[str, float]:
    """
    Args:
        prob_map: dataset name -> prob
        ref_loss: dataset name -> ref loss
        curr_loss: dataset name -> curr loss

    Returns:
        prob_map: updated prob map

    References:
        Dynamic Batch Loading in ShearedLlama (http://arxiv.org/abs/2310.06694)
    """
    task_types = [k for k in prob_map]
    original_weight = torch.tensor([prob_map[k] for k in task_types])
    diff = torch.tensor([curr_loss[k] - ref_loss[k] for k in task_types])
    eta = 1.0
    c = 1e-4

    updated_alpha = torch.log(original_weight) + eta * diff
    updated_alpha = torch.nn.functional.softmax(updated_alpha, dim=0)
    updated_domain_weights = (1 - c) * updated_alpha + c / len(task_types)
    updated_domain_weights = updated_domain_weights.detach().numpy().astype("float64")
    updated_domain_weights = updated_domain_weights / updated_domain_weights.sum()
    updated_domain_weights = updated_domain_weights.tolist()

    return {k: v for k, v in zip(task_types, updated_domain_weights)}


if __name__ == "__main__":
    # new_weight = update_weight_sheared_llama_paper(
    new_weight = update_weight_sheared_llama(
        LLAMA_DATA_PORTION,
        LLAMA2_7B_SLIMPAJAMA_VAL_REF_LOSS,
        {
            "en_book": 2.071,
            "en_wikipedia": 1.572,
            "en_stack": 1.491,
            "github": 0.705,
            "en_c4": 2.117,
            "en_cc": 1.728,
            "en_arxiv": 1.287,
        },
    )
    print(new_weight)
