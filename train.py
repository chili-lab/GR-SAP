import logging
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("Gloo").setLevel(logging.WARNING)
import argparse
import os
import pandas as pd
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from helper import model_mappings, format_query, format_response, total_num_examples
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

def print_config(args):
    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("\n")

def load_sft_dataset(dataset, num_examples):
    if dataset == 'gsm8k':
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.map(lambda x: {"messages": [{"role": "user", "content": format_query(x["question"], dataset)}, {"role": "assistant", "content": format_response(x["answer"], dataset)}]})
    elif dataset == 'math':
        # Load all math subsets and concatenate them
        math_configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
        math_datasets = []
        for config in math_configs:
            subset = load_dataset('EleutherAI/hendrycks_math', config, split='train')
            math_datasets.append(subset)
        ds = concatenate_datasets(math_datasets)
        ds = ds.map(lambda x: {"messages": [{"role": "user", "content": format_query(x["problem"], dataset)}, {"role": "assistant", "content": format_response(x["solution"], dataset)}]})
    elif dataset == 'alpaca_clean':
        ds = load_dataset("yahma/alpaca-cleaned", split='train')
        ds = ds.map(lambda x: {"messages": [{"role": "user", "content": format_query(x["instruction"], dataset, x["input"])}, {"role": "assistant", "content": format_response(x["output"], dataset)}]})
    elif dataset == 'hellaswag':
        ds = load_dataset("Rowan/hellaswag", split="train")
        ds = ds.map(lambda x: {"messages": [{"role": "user", "content": format_query(x["ctx"], dataset, options=x["endings"])}, {"role": "assistant", "content": format_response(x["label"], dataset)}]})
    elif dataset == 'winogrande':
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
        ds = ds.map(lambda x: {"messages": [{"role": "user", "content": format_query(x["sentence"], dataset, options=[x["option1"], x["option2"]])}, {"role": "assistant", "content": format_response(x["answer"], dataset)}]})
    elif dataset == 'medqa':
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        ds = ds.map(lambda x: {"messages": [{"role": "user", "content": format_query(x["question"], dataset, options=x["options"])}, {"role": "assistant", "content": format_response(x["answer_idx"], dataset)}]})
    elif dataset == 'sst2':
        ds = load_dataset("stanfordnlp/sst2", split="train")
        ds = ds.map(lambda x: {"messages": [{"role": "user", "content": format_query(x["sentence"], dataset)}, {"role": "assistant", "content": format_response(x["label"], dataset)}]})
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    assert num_examples <= len(ds), f"Requested {num_examples} examples, but dataset only has {len(ds)}"
    ds = ds.shuffle().select(range(num_examples))
    return ds

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model with DeepSpeed.")
    parser.add_argument(
        "--model",
        type=str,
        default="olmo2",
        help="Model to be fine-tuned;",
    )
    parser.add_argument(
        "--alignment_data_dir",
        type=str,
        default='./extraction_results',
        help="The directory containing alignment data files;",
    )
    parser.add_argument(
        "--alignment_data_file",
        type=str,
        default='extracted_qas_olmo2_128.csv',
        help="The extracted alignment data file;",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='gsm8k',
        choices=['gsm8k', 'math', 'hellaswag', 'winogrande', 'alpaca_clean', 'medqa', 'sst2'],
        help="The downstream fine-tuning dataset",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0,
        help="ratio of alignment data in the training data"
    )
    parser.add_argument(
        "--hard_ratio",
        type=float,
        default=0.5,
        help="ratio of hard examples in the alignment data (default: 0.5 for 1:1 hard:easy)"
    )
    parser.add_argument(
        "--training_method",
        type=str,
        default="full",
        choices=["full", "lora"],
        help="Training method: 'full' for full parameter fine-tuning, 'lora' for Low-Rank Adaptation."
    )
    parser.add_argument(
        "--control_type",
        type=str,
        default='total',
        help="control the total number or downstream dataset size",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=7168,
        help="total number of examples used for training per epoch"
    )

    parser.add_argument(
        "--num_save",
        type=int,
        default=1024,
        help="save the model trained on every N examples"
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size per device during training."
    )

    parser.add_argument(
        '--lr', 
        type=float, 
        default=2e-6,
        help="Learning rate."
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length."
    )

    parser.add_argument(
        "--deepspeed",
        type=str,
        required=True,
        help="Path to the DeepSpeed configuration file."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ckpt_output",
        help="Directory to save the trained model."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=0, 
        help="local_rank for distributed training on gpus"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    
    args = parser.parse_args()

    args.num_examples = total_num_examples[args.dataset]

    assert args.num_save % args.per_device_train_batch_size == 0, "num_save must be divisible by per_device_train_batch_size"
    assert 0 <= args.ratio <= 1, "ratio must be between 0 and 1"
    assert 0 <= args.hard_ratio <= 1, "hard_ratio must be between 0 and 1"

    print("Starting training with the following configuration:")
    print_config(args)

    model_id = model_mappings.get(args.model, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Fixed: use torch_dtype instead of dtype
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing for memory efficiency
    # model.gradient_checkpointing_enable()

    num_align = int(args.ratio * args.num_examples)
    if args.control_type == 'total':
        num_sft = args.num_examples - num_align
    elif args.control_type == 'downstream':
        num_sft = args.num_examples
    else:
        raise ValueError(f"Unsupported control_type: {args.control_type}")

    ds_sft = load_sft_dataset(args.dataset, num_sft)
    ds_sft = ds_sft.select_columns(["messages"])

    if args.ratio > 0:
        df_align = pd.read_csv(os.path.join(args.alignment_data_dir, args.alignment_data_file))
        # Filter out rows with NaN values in query or response
        df_align = df_align.dropna(subset=['query', 'response'])

        assert num_align <= len(df_align), f"Requested {num_align} alignment examples, but only {len(df_align)} available"

        # Filter for safe examples if is_safe column exists
        if 'is_safe' in df_align.columns:
            df_align = df_align[df_align['is_safe'] == True]
            print(f"Filtered to {len(df_align)} safe examples")
        
        # Sample hard cases first, then easy cases if difficulty column exists
        if 'difficulty' in df_align.columns:
            df_hard = df_align[df_align['difficulty'] == 'hard']
            df_easy = df_align[df_align['difficulty'] == 'easy']
            
            num_hard = min(len(df_hard), int(num_align * args.hard_ratio))
            num_easy = num_align - num_hard
            
            df_hard_sampled = df_hard.sample(n=num_hard, replace=False)
            df_easy_sampled = df_easy.sample(n=num_easy, replace=False)
            df_align = pd.concat([df_hard_sampled, df_easy_sampled], ignore_index=True)
            print(f"Sampled {num_hard} hard and {num_easy} easy cases (hard_ratio={args.hard_ratio})")
        else:
            # No difficulty column, sample randomly as before
            df_align = df_align.sample(n=num_align, replace=False)
        df_align["messages"] = df_align.apply(
            lambda row: [
                {"role": "user", "content": row["query"]},
                {"role": "assistant", "content": row["response"]}
            ],
            axis=1
        )
        ds_align = Dataset.from_pandas(df_align)
        ds_align = ds_align.select_columns(["messages"])
        train_dataset = concatenate_datasets([ds_sft, ds_align])
    else:
        train_dataset = ds_sft
    train_dataset = train_dataset.shuffle()

    model_name = model_id.split("/")[-1]
    alignment_data_name = os.path.splitext(os.path.basename(args.alignment_data_file))[0]
    output_dir = os.path.join(args.output_dir, f"{model_name}_{args.training_method}_{alignment_data_name}_{args.dataset}_lr{args.lr}_{args.control_type}_{args.num_examples}_{args.ratio}")
    os.makedirs(output_dir, exist_ok=True)

    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = args.per_device_train_batch_size * num_gpus * args.gradient_accumulation_steps
    if args.control_type == 'total':
        save_steps = int(args.num_save / global_batch_size)
    elif args.control_type == 'downstream':
        save_steps = int(args.num_save * (1 + args.ratio) / global_batch_size)
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.5},
        warmup_ratio=0.1,
        deepspeed=args.deepspeed,
        bf16=True,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        report_to="none",
        completion_only_loss=True,
        dataset_text_field="messages",
        max_length=args.max_length,
    )

    peft_config = None
    if args.training_method == "lora":
        rank = 16
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=2*rank,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Start fine-tuning...")
    trainer.train()

    # Save the final model
    output_dir_final = os.path.join(output_dir, "final_model")
    print(f"Saving model to {output_dir_final}")
    trainer.save_model(output_dir_final)
    print("Model saved successfully.")

    # Release all memory
    print("\nReleasing memory...")
    del trainer
    del model
    del tokenizer
    del train_dataset
    if 'ds_sft' in locals():
        del ds_sft
    if 'ds_align' in locals():
        del ds_align
    if 'df_align' in locals():
        del df_align
    if 'peft_config' in locals():
        del peft_config
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory cleared")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    print("Memory cleanup complete.")

if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.destroy_process_group()