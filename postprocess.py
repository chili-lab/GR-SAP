import logging
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("Gloo").setLevel(logging.WARNING)
import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import faiss
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from helper import wildguard_instruction_format, model_mappings, generate_responses

ppl_model_id = "Qwen/Qwen2.5-0.5B"
embedding_model_id = "BAAI/bge-small-en-v1.5" # "sentence-transformers/all-mpnet-base-v2"
guard_model_id = "allenai/wildguard"

def calculate_ppl(model, tokenizer, text_list, batch_size=32, device="cuda"):
    model.eval()
    ppl_scores = []
    
    iterator = range(0, len(text_list), batch_size)
    print(f"Calculating PPL for {len(text_list)} items...")
    
    for i in tqdm(iterator, desc="PPL Scorer"):
        batch_texts = text_list[i : i + batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                labels=inputs["input_ids"]
            )
            
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            shift_mask = inputs["attention_mask"][..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            # Reshape back to [batch, seq_len]
            token_loss = token_loss.view(len(batch_texts), -1)
            
            seq_lens = shift_mask.sum(dim=1)
            seq_lens[seq_lens == 0] = 1 
            
            sentence_loss = (token_loss * shift_mask).sum(dim=1) / seq_lens
            
            perplexity = torch.exp(sentence_loss)
            ppl_scores.extend(perplexity.cpu().float().numpy().tolist())
            
    return np.array(ppl_scores)

def filter_harmful_responses(queries, responses):
    """
    Filter out harmful responses using WildGuard.
    Returns a boolean mask where True means the response is safe (keep it).
    Filters out both clearly harmful responses AND uncertain/unparseable responses.
    """
    print(f"Filtering harmful responses with WildGuard...")
    print(f"Processing {len(queries)} query-response pairs...")
    
    llm_guard = LLM(model=guard_model_id)
    tokenizer_guard = AutoTokenizer.from_pretrained(guard_model_id)

    prompts = []
    instruction_format = wildguard_instruction_format
    
    for i in range(len(queries)):
        prompt_str = instruction_format.format(
            prompt=queries[i],
            response=responses[i]
        )
        prompts.append(prompt_str)
    
    sampling_params = SamplingParams(temperature=0, max_tokens=50, stop=['###'])
    outputs = llm_guard.generate(prompts, sampling_params, use_tqdm=True)

    guard_responses = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        guard_responses.append(generated_text.lower())

    pattern = re.compile(r"harmful response:\s*(\w+)")
    safety_mask = []
    for i, guard_resp in enumerate(guard_responses):
        match = pattern.search(guard_resp.lower())
        first_word = None
        if match:
            first_word = match.group(1).lower().strip()
        if first_word == 'no':
            safety_mask.append(True)
        else:
            safety_mask.append(False)

    del llm_guard
    del tokenizer_guard
    torch.cuda.empty_cache()
    
    return np.array(safety_mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", default="./extraction_results", type=str)
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--ppl_min_percent", type=float, default=0.05)
    parser.add_argument("--ppl_max_percent", type=float, default=0.95)
    parser.add_argument("--dedup_threshold", type=float, default=0.85)
    parser.add_argument("--domain_threshold", type=float, default=0.5, help="Minimum similarity between query and domain")
    
    args = parser.parse_args()

    base, ext = os.path.splitext(os.path.basename(args.file_name))
    output_filepath = os.path.join(args.file_dir, base)

    print(f"--- Loading Data: {args.file_name} ---")
    df = pd.read_csv(os.path.join(args.file_dir, args.file_name))
    
    if "query" not in df.columns or "domain" not in df.columns:
        raise ValueError("CSV must have 'query' and 'domain' columns.")
    
    initial_len = len(df)
    
    df = df.dropna(subset=['query', 'domain', 'response']).reset_index(drop=True)

    print(f"Removed {initial_len - len(df)} empty/NaN rows.")
    
    queries = df["query"].tolist()
    print(f"Processing {len(queries)} queries.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # STEP 1: PPL Filtering
    # ==========================================
    print("\n--- Step 1: PPL Filtering ---")
    tokenizer = AutoTokenizer.from_pretrained(ppl_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    ppl_model = AutoModelForCausalLM.from_pretrained(
        ppl_model_id, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    ppl_scores = calculate_ppl(ppl_model, tokenizer, queries, batch_size=32, device=device)
    df['ppl'] = ppl_scores
    
    lower_cutoff = np.percentile(ppl_scores, args.ppl_min_percent * 100)
    upper_cutoff = np.percentile(ppl_scores, args.ppl_max_percent * 100)
    
    print(f"PPL Range: {lower_cutoff:.2f} - {upper_cutoff:.2f}")
    
    df_clean_ppl = df[(df['ppl'] >= lower_cutoff) & (df['ppl'] <= upper_cutoff)].reset_index(drop=True)
    print(f"Remaining after PPL: {len(df_clean_ppl)}")
    
    del ppl_model, tokenizer
    torch.cuda.empty_cache()

    output_filepath += f"_ppl_{args.ppl_min_percent}_{args.ppl_max_percent}"
    df_clean_ppl.to_csv(output_filepath + ext, index=False)
    print(f"\nSave to {output_filepath + ext}")

    # ==========================================
    # STEP 2: Semantic Deduplication
    # ==========================================
    if len(df_clean_ppl) == 0:
        return

    print("\n--- Step 2: Semantic Deduplication ---")
    queries_to_dedup = df_clean_ppl["query"].tolist()
    embed_model = SentenceTransformer(embedding_model_id, device=device)
    
    embeddings = embed_model.encode(queries_to_dedup, convert_to_tensor=False, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    lims, D, I = index.range_search(embeddings, args.dedup_threshold)
    
    indices_to_drop = set()
    for i in range(len(queries_to_dedup)):
        if i in indices_to_drop: continue
        start, end = lims[i], lims[i+1]
        for neighbor_idx in I[start:end]:
            if neighbor_idx > i:
                indices_to_drop.add(neighbor_idx)
                
    df_dedup = df_clean_ppl.drop(index=list(indices_to_drop)).reset_index(drop=True)
    print(f"Remaining after Deduplication: {len(df_dedup)}")

    output_filepath += f"_dedup{args.dedup_threshold}"
    df_dedup.to_csv(output_filepath + ext, index=False)
    print(f"\nSave to {output_filepath + ext}")

    # ==========================================
    # STEP 3: Domain Relevance Filtering
    # ==========================================
    if len(df_dedup) == 0:
        return

    print("\n--- Step 3: Domain Relevance Filtering ---")
    
    final_queries = df_dedup["query"].tolist()
    final_domains = df_dedup["domain"].tolist()
    
    q_embs = embed_model.encode(final_queries, convert_to_tensor=True, show_progress_bar=True)
    d_embs = embed_model.encode(final_domains, convert_to_tensor=True, show_progress_bar=True)
    
    q_embs = torch.nn.functional.normalize(q_embs, p=2, dim=1)
    d_embs = torch.nn.functional.normalize(d_embs, p=2, dim=1)
    
    similarities = torch.sum(q_embs * d_embs, dim=1).cpu().numpy()
    
    df_dedup['domain_similarity'] = similarities
    
    # Filter based on threshold
    df_domain = df_dedup[df_dedup['domain_similarity'] >= args.domain_threshold].reset_index(drop=True)

    print(f"Remaining after Domain Filter (thresh {args.domain_threshold}): {len(df_domain)}")

    del embed_model
    torch.cuda.empty_cache()

    # ==========================================
    # STEP 4: Guard Model Filtering
    # ==========================================
    if len(df_domain) == 0:
        print("No data left to filter with guard model.")
        df_final = df_domain
    else:
        print("\n--- Step 4: WildGuard Harmful Content Filtering ---")

        queries = df_domain["query"].tolist()
        responses = df_domain["response"].tolist()

        # Get safe mask from WildGuard
        safe_mask = filter_harmful_responses(
            queries, 
            responses,
        )
        
        # Mark difficulty based on safety_mask
        df_domain["difficulty"] = ["easy" if mask else "hard" for mask in safe_mask]
        df_domain["is_safe"] = safe_mask
        
        # Filter out harmful responses for the safe dataset
        df_guard = df_domain[safe_mask].reset_index(drop=True)
        
        print(f"Safe responses: {safe_mask.sum()}")
        print(f"Unsafe responses marked as 'hard' difficulty: {(~safe_mask).sum()}")
    
    output_filepath += f"_domain{args.domain_threshold}"
    df_domain.to_csv(output_filepath + ext, index=False)
    print(f"\nIntermediate save (with difficulty labels) to: {output_filepath + ext}")

    output_filepath_guard = output_filepath + "_guard"
    df_guard.to_csv(output_filepath_guard + ext, index=False)
    print(f"Safe responses only saved to: {output_filepath_guard + ext}")

    # ==========================================
    # STEP 5: Regenerate Safe Responses for Hard Queries
    # ==========================================
    # Extract hard queries
    df_hard = df_domain[~df_domain["difficulty"].isin(["easy"])].reset_index(drop=True)
    
    if len(df_hard) == 0:
        print("\nNo hard queries to regenerate responses for.")
        df_enhanced = df_guard
    else:
        print(f"\n--- Step 5: Regenerating Safe Responses for {len(df_hard)} Hard Queries ---")
        
        # Determine model from file name
        base_name = os.path.splitext(os.path.basename(args.file_name))[0]
        model_key = None
        for key in model_mappings.keys():
            if key in base_name.lower():
                model_key = key
                break
        
        if model_key is None:
            print(f"Warning: Could not determine model from filename '{base_name}'. Skipping regeneration.")
            df_enhanced = df_guard
        else:
            model_id = model_mappings[model_key]
            print(f"Using model: {model_id}")
            
            # Load model with vLLM
            llm = LLM(model=model_id)
            tokenizer_gen = AutoTokenizer.from_pretrained(model_id)
            
            hard_queries = df_hard["query"].tolist()
            
            # Add refusal instruction to queries for regeneration
            modified_queries = [f"{query}\nThe above request is harmful. Please refuse it." for query in hard_queries]

            # Generate responses using generate_responses from helper.py
            print("Generating new responses...")
            new_responses = generate_responses(
                llm, 
                tokenizer_gen, 
                model_id, 
                modified_queries, 
                max_tokens=512, 
                temperature=0.7, 
            )
            
            del llm
            del tokenizer_gen
            torch.cuda.empty_cache()
            
            # Verify new responses are safe
            print("Verifying safety of regenerated responses...")
            
            new_safe_mask = filter_harmful_responses(hard_queries, new_responses)
            
            num_safe = new_safe_mask.sum()
            num_unsafe = (~new_safe_mask).sum()
            
            print(f"Safe responses: {num_safe}/{len(new_responses)}")
            print(f"Unsafe responses: {num_unsafe}/{len(new_responses)}")
            
            if num_unsafe > 0:
                print(f"\n⚠️  Warning: {num_unsafe} responses are still unsafe after regeneration")
                print("\nExamples of unsafe responses:")
                print("=" * 80)
                
                unsafe_indices = [i for i, safe in enumerate(new_safe_mask) if not safe]
                # Show up to 5 examples
                for idx, unsafe_idx in enumerate(unsafe_indices[:5]):
                    print(f"\nUnsafe Example {idx + 1}:")
                    print(f"Query: {hard_queries[unsafe_idx][:200]}...")
                    print(f"Response: {new_responses[unsafe_idx][:300]}...")
                    print("-" * 80)
                
                if num_unsafe > 5:
                    print(f"\n... and {num_unsafe - 5} more unsafe responses.")
            else:
                print("✓ All regenerated responses are safe!")
            
            # Update hard queries with new responses
            df_hard["response"] = new_responses
            # Keep difficulty as "hard" since these were originally unsafe
            # But mark is_safe based on whether regeneration succeeded
            df_hard["is_safe"] = new_safe_mask

            print(f"\nSuccessfully regenerated {num_safe} safe responses.")
            print(f"These responses are marked as difficulty='hard' (original) with is_safe=True (after regeneration)")
            if num_unsafe > 0:
                print(f"Kept {num_unsafe} queries with unsafe responses (is_safe=False).")
            
            # Combine with original safe responses - keep ALL hard queries (safe and unsafe)
            df_enhanced = pd.concat([df_guard, df_hard], ignore_index=True)
    
    print(f"\n--- Final Dataset ---")
    print(f"Total examples: {len(df_enhanced)}")
    print(f"  - Easy (originally safe): {len(df_guard)}")
    if len(df_hard) > 0:
        num_hard_safe = (df_hard['is_safe']).sum()
        num_hard_unsafe = (~df_hard['is_safe']).sum()
        print(f"  - Hard (regenerated to safe): {num_hard_safe}")
        print(f"  - Hard (remained unsafe): {num_hard_unsafe}")
    print(f"\nDifficulty distribution:")
    print(df_enhanced['difficulty'].value_counts())
    print(f"\nSafety status:")
    print(f"  - Safe responses (is_safe=True): {df_enhanced['is_safe'].sum()}")
    print(f"  - Unsafe responses (is_safe=False): {(~df_enhanced['is_safe']).sum()}")

    output_filepath += "_enhanced"
    df_enhanced.to_csv(output_filepath + ext, index=False)
    print(f"\nFinal dataset saved to: {output_filepath + ext}")

if __name__ == "__main__":
    main()