import logging
logging.getLogger("datasets").setLevel(logging.WARNING)
import argparse
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Extract safe response data from Nvidia Aegis AI Content Safety Dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./extraction_results",
        help="Directory to save the output CSV file"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="aegis_train.csv",
        help="Name of the output CSV file"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Nvidia Aegis AI Content Safety Dataset - Safe Response Extraction")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Output filename: {args.output_filename}")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Aegis dataset
    print(f"\nLoading Nvidia Aegis AI Content Safety Dataset 2.0...")
    try:
        dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split='train')
        print(f"Loaded {len(dataset)} total examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Convert to pandas DataFrame
    print("\nConverting to pandas DataFrame...")
    df = dataset.to_pandas()
    
    # Display available columns
    print(f"\nAvailable columns: {df.columns.tolist()}")
    
    # Display initial statistics
    print(f"\nInitial statistics:")
    print(f"  Total examples: {len(df)}")
    
    if 'response_label' in df.columns:
        print(f"\n  Response label distribution:")
        label_counts = df['response_label'].value_counts()
        for label, count in label_counts.items():
            print(f"    {label}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print("Error: 'response_label' column not found in dataset!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Filter for safe response data only
    print("\nFiltering for safe responses (response_label='safe')...")
    df_safe = df[df['response_label'] == 'safe'].copy()
    print(f"Filtered to {len(df_safe)} safe response examples")
    
    if len(df_safe) == 0:
        print("Warning: No safe response examples found!")
        return
    
    # Rename 'prompt' to 'query'
    print("\nRenaming 'prompt' column to 'query'...")
    if 'prompt' not in df_safe.columns:
        print("Error: 'prompt' column not found in dataset!")
        print(f"Available columns: {df_safe.columns.tolist()}")
        return
    
    df_safe = df_safe.rename(columns={'prompt': 'query'})
    
    # Reorder columns as specified
    print("\nSelecting and reordering columns...")
    desired_columns = [
        'prompt_label',
        'response_label',
        'prompt_label_source',
        'response_label_source',
        'violated_categories',
        'query',
        'response'
    ]
    
    # Keep only columns that exist in the dataframe
    existing_columns = [col for col in desired_columns if col in df_safe.columns]
    missing_columns = [col for col in desired_columns if col not in df_safe.columns]
    
    if missing_columns:
        print(f"Warning: Some requested columns are missing: {missing_columns}")
    
    # Keep only the specified columns (drop all others)
    df_safe = df_safe[existing_columns]
    
    print(f"Final column order: {df_safe.columns.tolist()}")
    
    # Drop rows with NaN values in query or response
    print("\nDropping rows with NaN values in 'query' or 'response' columns...")
    initial_count = len(df_safe)
    df_safe = df_safe.dropna(subset=['query', 'response'])
    dropped_count = initial_count - len(df_safe)
    print(f"Dropped {dropped_count} rows with NaN values ({dropped_count/initial_count*100:.2f}%)")
    print(f"Remaining examples: {len(df_safe)}")
    
    # Save to CSV
    output_path = os.path.join(args.output_dir, args.output_filename)
    print(f"\n{'='*80}")
    print(f"Saving to: {output_path}")
    df_safe.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_safe)} safe response examples to CSV")
    
    # Display statistics about the saved data
    print(f"\n{'='*80}")
    print("Final statistics:")
    print("=" * 80)
    print(f"  Total safe response examples saved: {len(df_safe)}")
    
    if 'prompt_label' in df_safe.columns:
        print(f"\n  Prompt label distribution:")
        prompt_label_counts = df_safe['prompt_label'].value_counts()
        for label, count in prompt_label_counts.items():
            print(f"    {label}: {count} ({count/len(df_safe)*100:.2f}%)")
    
    # Display sample
    print(f"\n{'='*80}")
    print("Sample of the first 3 rows:")
    print("=" * 80)
    for idx, row in df_safe.head(3).iterrows():
        print(f"\nExample {idx + 1}:")
        if 'prompt_label' in row:
            print(f"  Prompt label: {row['prompt_label']}")
        if 'response_label' in row:
            print(f"  Response label: {row['response_label']}")
        if 'query' in row:
            query_preview = str(row['query'])[:150]
            print(f"  Query: {query_preview}...")
        if 'response' in row:
            response_preview = str(row['response'])[:150]
            print(f"  Response: {response_preview}...")
    
    print(f"\n{'='*80}")
    print("Done!")
    print("=" * 80)

if __name__ == "__main__":
    main()
