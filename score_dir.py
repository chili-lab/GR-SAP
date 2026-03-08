import logging
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("Gloo").setLevel(logging.WARNING)
import os
import re
import torch
import argparse
import pandas as pd
from tabulate import tabulate
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from score import score_safety, compute_acc
from helper import *

def extract_number(s):
        match = re.search(r'\d+', s)
        if match:
            return int(match.group(0))
        else:
            return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--guard",
        type=str,
        default='wildguard',
        help="The guard model to be used;",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="The directory of model checkpoints",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        help="The number of examples used to train the model",
    )
    parser.add_argument(
        "--num_save",
        default=1024,
        type=int,
        help="Model is saved every N examples (checkpoint interval during training)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results/run3",
        help="The output directory to save results;",
    )

    args = parser.parse_args()

    if args.checkpoint_dir and args.checkpoint_dir.endswith('/'):
        args.checkpoint_dir = args.checkpoint_dir[:-1]
        
    for dataset in total_num_examples.keys():
        if dataset in args.checkpoint_dir.lower():
            args.num_examples = total_num_examples[dataset]

    training_method = "lora" if "lora" in args.checkpoint_dir.lower() else "full"
    
    base_model = None
    for model in model_mappings.keys():
        if model in args.checkpoint_dir.lower():
            base_model_id = model_mappings[model]
            print(f"Detected Base Model ID: {base_model_id}")
            break

    print(f"Guard Model: {args.guard}")
    print(f"num_examples: {args.num_examples}")
    print(f"num_save: {args.num_save}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"Detected Training Method: {training_method.upper()}")

    model_guard_id = get_model_guard_id(args.guard)
    checkpoints = [checkpoint for checkpoint in os.listdir(args.checkpoint_dir) if checkpoint.startswith("checkpoint-")]
    checkpoints_sorted = sorted(checkpoints, key=extract_number)
    # checkpoints_sorted = checkpoints_sorted[0:2]
    print(f"Find checkpoints: {checkpoints_sorted}")

    model_name_list = checkpoints_sorted[:-1] + ["final_model"]
    model_id_list = [os.path.join(args.checkpoint_dir, model_name) for model_name in model_name_list]
    num_examples_list = list(range(args.num_save, args.num_save * (len(checkpoints_sorted)), args.num_save)) + [args.num_examples]
    print(f"model_name_list: {model_name_list}")
    print(f"num_examples_list: {num_examples_list}")
    
    queries_test_dataset = {}
    for test_dataset in safety_test_datasets:
        df = load_safety_test_dataset(test_dataset) 
        queries = df['prompt'].tolist()
        # queries = queries[:50]
        queries_test_dataset[test_dataset] = queries

    reponses_test_dataset = {safety_test: [] for safety_test in safety_test_datasets}

    # Unified loop for both LoRA and Full fine-tuning evaluation
    for i, model_id in tqdm(enumerate(model_id_list), total=len(model_id_list)):
        
        if training_method == "lora":
            llm = LLM(model=base_model_id, enable_lora=True, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            lora_req = LoRARequest(f"adapter_{i}", i + 1, model_id)
        else:
            llm = LLM(model=model_id, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            lora_req = None

        for test_dataset, queries in queries_test_dataset.items():
            res = generate_responses(llm, tokenizer, model_id, queries, lora_request=lora_req, temperature=0)
            reponses_test_dataset[test_dataset].append(res)
        
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # for test_dataset in safety_test_datasets:
    #     test_dataset = load_test_dataset(test_dataset)
    #     queries = test_dataset['prompt'].tolist()
    #     quriries = queries[:50]

    #     all_responses = []
    #     for i, model_id in tqdm(enumerate(model_id_list), total=len(model_id_list)):
    #         print(f"Generating Response from model: {model_id}")
    #         llm = LLM(model=model_id)
    #         tokenizer = AutoTokenizer.from_pretrained(model_id)

    #         model_responses = generate_responses(llm, tokenizer, model_id, queries)
    #         all_responses.append(model_responses)
            
    #         del llm
    #         del tokenizer
    #         torch.cuda.empty_cache()
    #     reponses_test_dataset[test_dataset] = all_responses

    llm_guard = LLM(model=model_guard_id)
    tokenizer_guard = AutoTokenizer.from_pretrained(model_guard_id)

    scores_test_dataset = {}
    for test_dataset, all_responses in reponses_test_dataset.items():
        dataset_queries = queries_test_dataset[test_dataset]

        scores = []
        for i, model_responses in enumerate(all_responses):
            print(f"Scoring responses for test dataset: {test_dataset}")
            harmful_score, num_na_resp = score_safety(model_id_list[i], model_guard_id, llm_guard, tokenizer_guard, dataset_queries, model_responses, n_sample=20)
            scores.append(harmful_score)
        
        scores_test_dataset[test_dataset] = scores

    del llm_guard
    del tokenizer_guard
    torch.cuda.empty_cache()

    column_names = ["metric"] + num_examples_list
    
    data_rows = []
    for test_dataset, scores in scores_test_dataset.items():
        row = [test_dataset] + scores
        data_rows.append(row)

    # Add average row for safety test datasets
    if len(scores_test_dataset) > 0:
        avg_scores = []
        num_checkpoints = len(num_examples_list)
        for checkpoint_idx in range(num_checkpoints):
            checkpoint_scores = [scores_test_dataset[dataset][checkpoint_idx] for dataset in safety_test_datasets if dataset in scores_test_dataset]
            if checkpoint_scores:
                avg_score = sum(checkpoint_scores) / len(checkpoint_scores)
                avg_scores.append(avg_score)
            else:
                avg_scores.append(0)
        data_rows.append(["average_safety"] + avg_scores)

    # Compute accuracy for specific datasets
    computable_dataset = [d for d in downstream_datasets if d != 'alpaca_clean']
    for dataset in computable_dataset:
        if dataset in args.checkpoint_dir:
            acc_list = [compute_acc(model_id, dataset=dataset) for model_id in model_id_list]
            data_rows.append([f"accuracy_{dataset}"] + acc_list)
            break

    df = pd.DataFrame(data_rows, columns=column_names)

    csv_file_name = os.path.join(args.output_dir, f"{args.checkpoint_dir.split('/')[-1]}.csv")
    df.to_csv(csv_file_name, index=False, encoding='utf-8')

    df_print = df.copy()
    df_print.columns = [f"{col} (%)" for col in df.columns]

    print(tabulate(df_print, headers='keys', tablefmt='psql', showindex=False))

if __name__ == "__main__":
    main()