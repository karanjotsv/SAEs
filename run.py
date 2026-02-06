import os
import time
import json
import random
import argparse
import itertools
from dotenv import load_dotenv

import torch as t
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import config as dataset_config

import dictionary_learning.utils as utils
from dictionary_learning.evaluation import evaluate

import config
from train import run_training, run_evaluation


load_dotenv()
hf_token = os.getenv("HF_TOKEN") 

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir", type=str, required=True, help="where to store sweep"
    )
    parser.add_argument(
        "--dataset_name", type=str, help="what dataset to train SAE on"
    )
    parser.add_argument("--use_wandb", action="store_true", help="use wandb logging")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument(
        "--save_checkpoints", action="store_true", help="save checkpoints"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", help="layers to train SAE on"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="which language model to use"
    )
    parser.add_argument(
        "--architectures", type=str, nargs="+", choices=[e.value for e in config.TrainerType], required=True, help="which SAE architectures to train"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device to train on"
    )
    parser.add_argument(
        "--mixed_dataset", action="store_true", help="use mixed dataset"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=0 python run.py --save_dir run --dataset_name alpaca --model_id google/gemma-2-2b --layers 12 --architectures standard
    python run.py --save_dir run1 --model_id EleutherAI/pythia-70m-deduped --layers 3 --architectures standard jump_relu batch_top_k top_k gated --use_wandb
    python run.py --save_dir jumprelu --model_id EleutherAI/pythia-70m-deduped --layers 3 --architectures jump_relu --use_wandb
    """
    args = get_args()

    # prevents random CUDA out of memory errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # for wandb to work with multiprocessing
    mp.set_start_method("spawn", force=True)

    # when internet issues on cloud GPUs and then the streaming read fails
    # hopefully the outage is shorter than 100 * 20 seconds
    dataset_config.STREAMING_READ_MAX_RETRIES = 100
    dataset_config.STREAMING_READ_RETRY_INTERVAL = 20

    start_time = time.time()

    save_dir = (
        f"{args.save_dir}_{args.model_id}_{'_'.join(args.architectures)}".replace(
            "/", "_"
        )
    )
    if args.layers:
        for layer in args.layers:
            run_training(
                model_id=args.model_id,
                layer=layer,
                dataset_name=args.dataset_name,
                save_dir=save_dir,
                device=args.device,
                architectures=args.architectures,
                num_tokens=config.num_tokens,
                random_seeds=config.random_seeds,
                dictionary_widths=config.dictionary_widths,
                learning_rates=config.learning_rates,
                dry_run=args.dry_run,
                use_wandb=args.use_wandb,
                save_checkpoints=args.save_checkpoints,
                hf_token=hf_token,
            )
    else:
        run_training(
                model_id=args.model_id,
                save_dir=save_dir,
                dataset_name=args.dataset_name,
                device=args.device,
                architectures=args.architectures,
                num_tokens=config.num_tokens,
                random_seeds=config.random_seeds,
                dictionary_widths=config.dictionary_widths,
                learning_rates=config.learning_rates,
                dry_run=args.dry_run,
                use_wandb=args.use_wandb,
                save_checkpoints=args.save_checkpoints,
                hf_token=hf_token,
            )

    sae_paths = utils.get_nested_folders(save_dir)

    run_evaluation(
        args.model_id,
        sae_paths,
        args.dataset_name,
        config.eval_num_inputs,
        args.device,
        overwrite_prev_results=True,
        hf_token=hf_token,
        config=config
    )

    print(f"total time: {time.time() - start_time}")

    # python run.py --save_dir run --model_id google/gemma-2-2b --architectures standard
