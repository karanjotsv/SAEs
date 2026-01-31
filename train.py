import os
import json
import random

import torch as t
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import config as dataset_config

import dictionary_learning.utils as utils
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.pytorch_buffer import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate

import config
from config import remove_bos


def run_training(
    model_id: str,
    save_dir: str,
    device: str,
    architectures: list,
    num_tokens: int,
    random_seeds: list[int],
    dictionary_widths: list[int],
    learning_rates: list[float],
    hf_token: str,
    layer: int = None,
    dry_run: bool = False,
    use_wandb: bool = False,
    save_checkpoints: bool = False,
    buffer_tokens: int = 250_000,
):
    """
    Train SAE models on the given dataset.

    Args:
        model_id: str, the model id to use for training
        layer: int, the layer to train the SAE model on
        save_dir: str, where to store the sweep
        device: str, the device to train on
        architectures: list, the SAE architectures to train
        num_tokens: int, the number of tokens to train on
        random_seeds: list[int], the random seeds to use for training
        dictionary_widths: list[int], the widths of the dictionaries to use
        learning_rates: list[float], the learning rates to use
        dry_run: bool, whether to dry run the sweep (default=False)
        use_wandb: bool, whether to use wandb logging (default=False)
        save_checkpoints: bool, whether to save checkpoints (default=False)
        buffer_tokens: int, the number of tokens to store in the buffer (default=250_000)

    Returns:
        None
    """
    random.seed(config.random_seeds[0])
    t.manual_seed(config.random_seeds[0])

    # model and data parameters
    context_length = config.LLM_CONFIG[model_id].context_length

    llm_batch_size = config.LLM_CONFIG[model_id].llm_batch_size
    sae_batch_size = config.LLM_CONFIG[model_id].sae_batch_size
    dtype = config.LLM_CONFIG[model_id].dtype

    num_buffer_inputs = buffer_tokens // context_length
    print(f"buffer_size: {num_buffer_inputs}, buffer_size_in_tokens: {buffer_tokens}")
     # log the training on wandb or print to console every log_steps
    log_steps = 100 
    # number of batches to train
    steps = int(num_tokens / sae_batch_size) 

    if save_checkpoints:
        # checkpoints at 0.0%, 0.1%, 0.316%, 1%, 3.16%, 10%, 31.6%, 100% of training
        desired_checkpoints = t.logspace(-3, 0, 7).tolist()
        desired_checkpoints = [0.0] + desired_checkpoints[:-1]
        desired_checkpoints.sort()
        print(f"desired_checkpoints: {desired_checkpoints}")

        save_steps = [int(steps * step) for step in desired_checkpoints]
        save_steps.sort()
        print(f"save_steps: {save_steps}")
    else:
        save_steps = None

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=hf_token, dtype=dtype)
    # if layer = 12:, model = model[ : layer + 1]
    model = utils.truncate_model(model, layer)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    # if layer = 12:, submodel = model[layer]
    submodule, layer = utils.get_submodule(model, layer)
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    if "Qwen" in model_id and remove_bos:
        print(
            "\n\nWARNING: Qwen models do not have a bos token, we will remove the first non-pad token"
        )
    ###
    generator = hf_dataset_to_generator(
        "cornell-movie-review-data/rotten_tomatoes",
    )
    ###
    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=num_buffer_inputs,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=device,
        add_special_tokens=False,
        remove_bos=remove_bos,
        max_activation_norm_multiple=config.max_activation_norm_multiple,
    )

    trainer_configs = config.get_trainer_configs(
        architectures,
        learning_rates,
        random_seeds,
        activation_dim,
        dictionary_widths,
        model_id,
        device,
        layer,
        submodule_name,
        steps,
    )

    print(f"len trainer configs: {len(trainer_configs)}")
    assert len(trainer_configs) > 0
    save_dir = f"{save_dir}/{submodule_name}"

    if not dry_run:
        # run sweep
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            use_wandb=use_wandb,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            wandb_project=config.wandb_project,
            # normalize_activations=True,
            verbose=False,
            autocast_dtype=t.bfloat16,
            backup_steps=1000,
        )


@t.no_grad()
def run_evaluation(
    model_id: str,
    sae_paths: list[str],
    n_inputs: int,
    device: str,
    hf_token: str,
    config,
    overwrite_prev_results: bool = True,
    transcoder: bool = False,
):
    """
    Evaluate the given SAE models on the given inputs.

    Args:
        model_id: str, the model id to use for evaluation
        sae_paths: list[str], the paths to the SAE models to evaluate
        n_inputs: int, the number of inputs to evaluate
        device: str, the device to use for evaluation
        overwrite_prev_results: bool, whether to overwrite any existing eval results
        transcoder: bool, whether to use the transcoder model for evaluation

    Returns:
        dict, the evaluation results
    """
    random.seed(config.random_seeds[0])
    t.manual_seed(config.random_seeds[0])

    if transcoder:
        io = "in_and_out"
    else:
        io = "out"

    context_length = config.LLM_CONFIG[model_id].context_length
    llm_batch_size = config.LLM_CONFIG[model_id].llm_batch_size
    loss_recovered_batch_size = max(llm_batch_size // 5, 1)
    sae_batch_size = loss_recovered_batch_size * context_length
    dtype = config.LLM_CONFIG[model_id].dtype

    max_layer = 0

    for sae_path in sae_paths:
        config_path = f"{sae_path}/config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        layer = config["trainer"]["layer"]
        max_layer = max(max_layer, layer)

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=hf_token, dtype=dtype)
    model = utils.truncate_model(model, max_layer)

    buffer_size = n_inputs
    io = "out"
    n_batches = n_inputs // loss_recovered_batch_size

    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")

    input_strings = []
    for i, example in enumerate(generator):
        input_strings.append(example)
        if i > n_inputs * 5 * llm_batch_size:
            break

    eval_results = {}

    for sae_path in sae_paths:
        output_filename = f"{sae_path}/eval_results.json"
        if not overwrite_prev_results:
            if os.path.exists(output_filename):
                print(f"skipping {sae_path} as eval results already exist")
                continue

        dictionary, config = utils.load_dictionary(sae_path, device)
        dictionary = dictionary.to(dtype=model.dtype)

        layer = config["trainer"]["layer"]
        submodule, _ = utils.get_submodule(model, layer)

        activation_dim = config["trainer"]["activation_dim"]

        activation_buffer = ActivationBuffer(
            iter(input_strings),
            model,
            submodule,
            n_ctxs=buffer_size,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=sae_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=device,
            remove_bos=remove_bos,
            # max_activation_norm_multiple=config.max_activation_norm_multiple,
        )

        eval_results = evaluate(
            dictionary,
            activation_buffer,
            context_length,
            loss_recovered_batch_size,
            io=io,
            device=device,
            n_batches=n_batches,
        )

        hyperparameters = {
            "n_inputs": n_inputs,
            "context_length": context_length,
        }
        eval_results["hyperparameters"] = hyperparameters

        print(eval_results)

        with open(output_filename, "w") as f:
            json.dump(eval_results, f)

    # return the final eval_results for testing
    return eval_results
