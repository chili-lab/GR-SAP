import logging
logging.getLogger("datasets").setLevel(logging.WARNING)
import argparse
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Mapping of subdataset names to their sources in the Tulu-3 mixture
SUBDATASET_SOURCES = {
    "coconot": "ai2-adapt-dev/coconot_converted",
    "wildguard": "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
    "wildjailbreak": "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
}

def extract_query_response(messages):
    """
    Extract query (first user message) and response (first assistant message) from messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        tuple: (query, response)
    """
    query = ""
    response = ""
    
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        
        if role == 'user' and not query:
            query = content
        elif role == 'assistant' and not response:
            response = content
            
    return query, response

def main():
    parser = argparse.ArgumentParser(description="Extract subdatasets from Tulu-3 mixture and save to CSV")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./extraction_results",
        help="Directory to save the output CSV file"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="qa_olmo_original_tulu3.csv",
        help="Name of the output CSV file"
    )
    parser.add_argument(
        "--subdatasets",
        type=str,
        nargs="+",
        default=["coconot", "wildguard", "wildjailbreak"],
        choices=["coconot", "wildguard", "wildjailbreak"],
        help="Which subdatasets to extract (default: all three)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to extract per subdataset (default: all)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Tulu-3 Subdataset Extraction")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Output filename: {args.output_filename}")
    print(f"Subdatasets to extract: {', '.join(args.subdatasets)}")
    if args.max_examples:
        print(f"Max examples per subdataset: {args.max_examples}")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the full Tulu-3 mixture dataset
    print("\nLoading Tulu-3 SFT mixture dataset...")
    dataset_tulu = load_dataset("allenai/tulu-3-sft-olmo-2-mixture", split='train')
    df_tulu = dataset_tulu.to_pandas()
    print(f"Loaded {len(df_tulu)} total examples from Tulu-3 mixture")
    
    # List to collect dataframes from each subdataset
    all_dfs = []
    
    # Process each requested subdataset
    for subdataset_name in args.subdatasets:
        source = SUBDATASET_SOURCES[subdataset_name]
        print(f"\n{'='*80}")
        print(f"Processing: {subdataset_name}")
        print(f"Source: {source}")
        print(f"{'='*80}")
        
        # Filter by source
        df_subset = df_tulu[df_tulu['source'] == source].copy()
        print(f"Found {len(df_subset)} examples")
        
        if len(df_subset) == 0:
            print(f"Warning: No examples found for {subdataset_name}")
            continue
        
        # Apply max_examples limit if specified
        if args.max_examples and len(df_subset) > args.max_examples:
            print(f"Sampling {args.max_examples} examples...")
            df_subset = df_subset.sample(n=args.max_examples, random_state=42)
        
        # Extract query and response from messages
        print("Extracting query and response from messages...")
        tqdm.pandas(desc=f"Processing {subdataset_name}")
        df_subset[['query', 'response']] = df_subset['messages'].progress_apply(
            lambda x: pd.Series(extract_query_response(x))
        )
        
        # Keep only source, query, and response columns
        df_subset = df_subset[['source', 'query', 'response']]
        
        # Remove any rows with empty query or response
        initial_len = len(df_subset)
        df_subset = df_subset[
            (df_subset['query'].str.strip() != '') & 
            (df_subset['response'].str.strip() != '')
        ]
        final_len = len(df_subset)
        
        if initial_len != final_len:
            print(f"Removed {initial_len - final_len} examples with empty query or response")
        
        print(f"Final count for {subdataset_name}: {len(df_subset)} examples")
        
        # Add to collection
        all_dfs.append(df_subset)
    
    # Combine all subdatasets
    if not all_dfs:
        print("\nError: No data extracted from any subdataset!")
        return
    
    print(f"\n{'='*80}")
    print("Combining subdatasets...")
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total combined examples: {len(df_combined)}")
    
    # Display statistics
    print(f"\n{'='*80}")
    print("Statistics by source:")
    print("=" * 80)
    source_counts = df_combined['source'].value_counts()
    for source, count in source_counts.items():
        subdataset_name = [k for k, v in SUBDATASET_SOURCES.items() if v == source][0]
        print(f"  {subdataset_name:15s}: {count:6d} examples")
    
    # Save to CSV
    output_path = os.path.join(args.output_dir, args.output_filename)
    print(f"\n{'='*80}")
    print(f"Saving to: {output_path}")
    df_combined.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_combined)} examples to CSV")
    
    # Display sample
    print(f"\n{'='*80}")
    print("Sample of the first 3 rows:")
    print("=" * 80)
    for idx, row in df_combined.head(3).iterrows():
        subdataset_name = [k for k, v in SUBDATASET_SOURCES.items() if v == row['source']][0]
        print(f"\nExample {idx + 1} ({subdataset_name}):")
        print(f"  Query: {row['query'][:100]}...")
        print(f"  Response: {row['response'][:100]}...")
    
    print(f"\n{'='*80}")
    print("Done!")
    print("=" * 80)

if __name__ == "__main__":
    main()
