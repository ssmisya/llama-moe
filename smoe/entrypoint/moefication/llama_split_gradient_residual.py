import argparse
import os

from tqdm import tqdm
from transformers import LlamaConfig

from smoe.utils.io import torch_load_template_score_file
from smoe.utils.moefication.expert_split_residual import GradientSplitResidual
from smoe.utils.string_operation import str2bool

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--score_file_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--visualization_path', type=str, default=None)
    parser.add_argument('--expert_num_moe', type=int)
    parser.add_argument('--expert_num_residual', type=int)
    parser.add_argument('--expert_size', type=int)
    parser.add_argument('--template', type=str, default='layers.{}.mlp.gate_proj.weight')

    parser.add_argument('--kernel', type=str, default="plain", choices=("plain", "l1_norm", "l2_norm"))
    parser.add_argument('--accumulate_level', type=str, default="sample", choices=("sample", "total"))
    parser.add_argument('--criterion', type=str, default="min", choices=("min", "max"))
    parser.add_argument('--importance_type', type=str, default="feature_grad", choices=("feature_grad", "feature_change"))

    args = parser.parse_args()
    print(args, "\n")

    print("Loading llama config...")
    config = LlamaConfig.from_pretrained(args.model_path)

    print("Processing layers...")
    save_root_path = args.save_path

    if args.importance_type == "feature_grad":
        file_postfix = ".grad"
    elif args.importance_type == "feature_change":
        file_postfix = ".change"
    else:
        raise NotImplementedError

    for i in tqdm(range(config.num_hidden_layers)):
        score_list = torch_load_template_score_file(args.score_file_path, args.template + file_postfix, i)

        args.save_path = os.path.join(
            save_root_path,
            f"{os.path.split(args.model_path)[1]}-Split-Gradient-{args.criterion}-{args.kernel}-{args.accumulate_level}-{args.importance_type}",
            f"{args.expert_num_moe}Experts-{args.expert_num_residual}Residuals-{args.expert_size}Neurons-Share"
        )

        split = GradientSplitResidual(args, args.template, i, score_list)
        split.split(args.expert_num_moe, args.expert_num_residual, args.expert_size, criterion=args.criterion)
        split.save()

        if args.visualization_path is not None:
            split.visualize(args.visualization_path)
    print("Done.")
    # fmt: on
