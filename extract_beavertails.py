import logging
logging.getLogger("datasets").setLevel(logging.WARNING)
import argparse
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Extract safe data from BeaverTails dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./extraction_results",
        help="Directory to save the output CSV file"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="beavertails_30k_train.csv",
        help="Name of the output CSV file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="30k_train",
        choices=["30k_train", "30k_test", "330k_train", "330k_test"],
        help="Which split to load (default: 30k_train)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BeaverTails Safe Data Extraction")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Output filename: {args.output_filename}")
    print(f"Split: {args.split}")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load BeaverTails dataset
    print(f"\nLoading BeaverTails dataset (split: {args.split})...")
    dataset = load_dataset("PKU-Alignment/BeaverTails", split=args.split)
    print(f"Loaded {len(dataset)} total examples")
    
    # Convert to pandas DataFrame
    print("\nConverting to pandas DataFrame...")
    df = dataset.to_pandas()
    
    # Display initial statistics
    print(f"\nInitial statistics:")
    print(f"  Total examples: {len(df)}")
    if 'is_safe' in df.columns:
        safe_count = df['is_safe'].sum()
        unsafe_count = len(df) - safe_count
        print(f"  Safe examples: {safe_count} ({safe_count/len(df)*100:.2f}%)")
        print(f"  Unsafe examples: {unsafe_count} ({unsafe_count/len(df)*100:.2f}%)")
    
    # Filter for safe data only
    print("\nFiltering for safe data (is_safe=True)...")
    if 'is_safe' not in df.columns:
        print("Error: 'is_safe' column not found in dataset!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    df_safe = df[df['is_safe'] == True].copy()
    print(f"Filtered to {len(df_safe)} safe examples")
    
    if len(df_safe) == 0:
        print("Warning: No safe examples found!")
        return
    
    # Rename 'prompt' to 'query'
    print("\nRenaming 'prompt' column to 'query'...")
    if 'prompt' not in df_safe.columns:
        print("Error: 'prompt' column not found in dataset!")
        print(f"Available columns: {df_safe.columns.tolist()}")
        return
    
    df_safe = df_safe.rename(columns={'prompt': 'query'})
    
    # Reorder columns
    print("\nReordering columns...")
    column_order = ['category', 'is_safe', 'query', 'response']
    # Keep only columns that exist in the dataframe
    existing_columns = [col for col in column_order if col in df_safe.columns]
    # Add any remaining columns that weren't in our specified order
    remaining_columns = [col for col in df_safe.columns if col not in existing_columns]
    final_column_order = existing_columns + remaining_columns
    df_safe = df_safe[final_column_order]
    
    # Display available columns
    print(f"\nColumns in the dataset (in order): {df_safe.columns.tolist()}")
    
    # Check for common columns
    key_columns = []
    if 'category' in df_safe.columns:
        key_columns.append('category')
    if 'is_safe' in df_safe.columns:
        key_columns.append('is_safe')
    if 'query' in df_safe.columns:
        key_columns.append('query')
    if 'response' in df_safe.columns:
        key_columns.append('response')

    print(f"Key columns identified: {key_columns}")
    
    # Save to CSV
    output_path = os.path.join(args.output_dir, args.output_filename)
    print(f"\n{'='*80}")
    print(f"Saving to: {output_path}")
    df_safe.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_safe)} safe examples to CSV")
    
    # Display statistics about the saved data
    print(f"\n{'='*80}")
    print("Final statistics:")
    print("=" * 80)
    print(f"  Total safe examples saved: {len(df_safe)}")
    
    if 'category' in df_safe.columns:
        print(f"\n  Distribution by category:")
        category_counts = df_safe['category'].value_counts()
        for category, count in category_counts.head(10).items():
            print(f"    {str(category):<50s}: {count:6d}")
    
    # Display sample
    print(f"\n{'='*80}")
    print("Sample of the first 3 rows:")
    print("=" * 80)
    for idx, row in df_safe.head(3).iterrows():
        print(f"\nExample {idx + 1}:")
        if 'query' in row:
            query_preview = str(row['query'])[:150]
            print(f"  Query: {query_preview}...")
        if 'response' in row:
            response_preview = str(row['response'])[:150]
            print(f"  Response: {response_preview}...")
        if 'category' in row:
            print(f"  Category: {row['category']}")
    
    print(f"\n{'='*80}")
    print("Done!")
    print("=" * 80)

if __name__ == "__main__":
    main()
