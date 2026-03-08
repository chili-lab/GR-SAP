import logging
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("Gloo").setLevel(logging.WARNING)
import os
import sys
import argparse
import math
import mauve
import pandas as pd
import time
import numpy as np
import gc
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.inputs import TokensPrompt
from tabulate import tabulate

dataset_ori_mappings = {
    "pku-beavertails": "PKU-Alignment/BeaverTails",
    "pku-saferlhf": "PKU-Alignment/PKU-SafeRLHF",
    "aegis": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
    "GR-SAP": "./extraction_results/qa_olmo_512.csv"
}

def process_source_tulu(messages):
    combined_text = ""
    extracted_query = ""
    query_found = False
    
    for message in messages:
        role = message['role']
        content = message['content']
        
        combined_text += f"{role.upper()}: {content}\n"

        if role == 'user' and not query_found:
            extracted_query = content
            query_found = True
            
    combined_text = combined_text.strip()  # Remove trailing newline
    return extracted_query, combined_text

def process_target(query, response):
    # Combine prompt and response into a single text
    combined_text = "USER: " + query + "\n" + "ASSISTANT: " + response
    return combined_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        type=str,
        default='mauve',
        help="Similarity metric to use: mauve",
    )

    parser.add_argument(
        "--file_dir",
        type=str,
        default='./extraction_results',
        help="The test file directory",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='../similarity_results',
        help="The output file directory",
    )

    parser.add_argument(
        "--file_name",
        type=str,
        help="The test file name",
    )

    parser.add_argument(
        "--example",
        type=int,
        default=10000,
        help="The example number to use",
    )

    parser.add_argument(
        "--text_type",
        type=str,
        default="query",
        help="Type of text to compare: query, response or both",
    )

    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Run number for mauve computation",
    )

    parser.add_argument(
        "--m_batch_size",
        type=int,
        default=64,
        help="Batch size for mauve computation",
    )

    parser.add_argument(
        "--mauve_scaling_factor",
        type=float,
        default=2.0,
        help="Scaling factor for MAUVE computation (controls sensitivity to distribution differences)",
    )

    args = parser.parse_args()
    
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    df_target = pd.read_csv(os.path.join(args.file_dir, args.file_name))
    df_target = df_target.dropna(subset=['query', 'response']).reset_index(drop=True)

    df_target['query'] = df_target['query'].astype(str)
    df_target['response'] = df_target['response'].astype(str)
    df_target = df_target.sample(n=min(args.example, len(df_target))).reset_index(drop=True)

    df_target['processed_response'] = df_target.apply(lambda row: process_target(row['query'], row['response']), axis=1)

    dataset_tulu = load_dataset("allenai/tulu-3-sft-olmo-2-mixture", split='train')
    df_tulu = dataset_tulu.to_pandas()

    score_results = {}
    for dataset_name, source in dataset_ori_mappings.items():
        if 'tulu' in dataset_name:
            df_source = df_tulu[df_tulu['source']==source]
            num_example = min(args.example, len(df_source))
            print(f"Using {num_example} examples from source: {source}")
            df_source = df_source.sample(n=num_example).reset_index(drop=True)
            df_source[['query', 'processed_response']] = df_source['messages'].apply(process_source_tulu).tolist()
        elif source.endswith('.csv'):
            # Handle CSV files directly
            df_source = pd.read_csv(source)
            df_source = df_source.dropna(subset=['query', 'response']).reset_index(drop=True)
            df_source['query'] = df_source['query'].astype(str)
            df_source['response'] = df_source['response'].astype(str)
            num_example = min(args.example, len(df_source))
            print(f"Using {num_example} examples from source: {source}")
            df_source = df_source.sample(n=num_example).reset_index(drop=True)
            df_source['processed_response'] = df_source.apply(lambda row: process_target(row['query'], row['response']), axis=1)
        else:
            if "aegis" in dataset_name:
                dataset_source = load_dataset(source, split='train')
                dataset_source = dataset_source.filter(lambda example: example["response_label"] == "safe")
            if "beavertails" in dataset_name:
                dataset_source = load_dataset(source, split='330k_train')
                dataset_source = dataset_source.filter(lambda example: example["is_safe"] == True)
            elif "saferlhf" in dataset_name:
                dataset_source = load_dataset(source, split='train')
                
                dataset_source = dataset_source.filter(lambda example: example["prompt_source"] != "Beavertails")
                
                def select_safer_response(example):
                    if example["safer_response_id"] == 0:
                        return {"response": example["response_0"]}
                    else:
                        return {"response": example["response_1"]}
                
                dataset_source = dataset_source.map(select_safer_response)
            
            df_source = dataset_source.to_pandas()
            num_example = min(args.example, len(df_source))
            print(f"Using {num_example} examples from source: {source}")
            df_source = df_source.sample(n=num_example).reset_index(drop=True)
            df_source['query'] = df_source['prompt']
            df_source['processed_response'] = df_source.apply(lambda row: process_target(row['query'], row['response']), axis=1)

        querys_target = df_target['query'].tolist()
        querys_source = df_source['query'].tolist()

        response_target = df_target['processed_response'].tolist()
        response_source = df_source['processed_response'].tolist()

        if args.metric == 'mauve':
            start_time = time.perf_counter()

            if args.text_type == 'query':
                out = mauve.compute_mauve(p_text=querys_source, q_text=querys_target, batch_size=args.m_batch_size, device_id=0, mauve_scaling_factor=args.mauve_scaling_factor)
                score_results[dataset_name] = {'query_mauve': out.mauve}
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Time taken to compute MAUVE score for {dataset_name} (query): {elapsed_time:.2f} seconds")
            elif args.text_type == 'response':
                out = mauve.compute_mauve(p_text=response_source, q_text=response_target, batch_size=args.m_batch_size, device_id=0, mauve_scaling_factor=args.mauve_scaling_factor)
                score_results[dataset_name] = {'response_mauve': out.mauve}
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Time taken to compute MAUVE score for {dataset_name} (response): {elapsed_time:.2f} seconds")
            else:  # both
                out_query = mauve.compute_mauve(p_text=querys_source, q_text=querys_target, batch_size=args.m_batch_size, device_id=0, mauve_scaling_factor=args.mauve_scaling_factor)
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                out_response = mauve.compute_mauve(p_text=response_source, q_text=response_target, batch_size=args.m_batch_size, device_id=0, mauve_scaling_factor=args.mauve_scaling_factor)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Time taken to compute MAUVE score for {dataset_name} (both): {elapsed_time:.2f} seconds")
                
                score_results[dataset_name] = {
                    'query_mauve': out_query.mauve,
                    'response_mauve': out_response.mauve
                }
        
        # Clean up GPU memory after processing each dataset
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"GPU memory cleared after processing {dataset_name}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    df_results = pd.DataFrame.from_dict(score_results, orient='index')
    
    df_results.index.name = 'dataset_name'
    df_results.reset_index(inplace=True)

    filename = os.path.basename(args.file_name)
    base, ext = os.path.splitext(filename)
    csv_name = os.path.join(args.output_dir, f"{base}_{args.text_type}_similarity_{args.run}.csv")
    print(f"\nSaving results to {csv_name}...")
    df_results.to_csv(csv_name, index=False)
    
    print(f"File Path: {args.file_name}")
    
    df_display = df_results.set_index('dataset_name').T.reset_index()
    df_display.rename(columns={'index': 'Metric'}, inplace=True)
    
    print("\n" + tabulate(df_display, headers='keys', tablefmt="grid", floatfmt=".4f", showindex=False))

if __name__ == "__main__":
    main()