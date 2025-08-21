import pandas as pd
import numpy as np
import json
from itertools import combinations
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

# --- Configuration ---
RAW_DATA_PATH = Path("data/asap-sas/train.tsv")
PROCESSED_DATA_DIR = Path("processed_data")
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
PAIRS_FILE = PROCESSED_DATA_DIR / "train_pairs.jsonl"

def prepare_dataset(val_size=0.15, test_size=0.15):
    """
    Loads, cleans, splits, and saves the ASAP-SAS dataset.
    The split is done based on 'EssaySet' to prevent data leakage.
    Also generates and saves pairs of answers for pairwise ranking loss.
    """
    print("--- Starting Data Preparation ---")
    
    # Create directory for processed data
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH, sep='\t', encoding='ISO-8859-1')
    df = df.rename(columns={"EssayText": "answer", "EssaySet": "prompt_id"})
    
    # 2. Clean Data & Create Target Score
    df = df.dropna(subset=['answer'])
    df['score'] = (df['Score1'] + df['Score2']) / 2.0
    
    # 3. Split Data by EssaySet (prompt_id) to prevent leakage
    print("Splitting data into train, validation, and test sets...")
    prompts = df['prompt_id'].unique()
    
    # Split prompts for train/val and test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['prompt_id']))
    
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    # Adjust validation size relative to the remaining data
    val_size_adjusted = val_size / (1 - test_size)
    
    # Split prompts for train and val
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=42)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['prompt_id']))
    
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]
    
    print(f"Train set size: {len(train_df)} ({len(train_df['prompt_id'].unique())} prompts)")
    print(f"Validation set size: {len(val_df)} ({len(val_df['prompt_id'].unique())} prompts)")
    print(f"Test set size: {len(test_df)} ({len(test_df['prompt_id'].unique())} prompts)")

    # 4. Generate Pairwise Data for Training
    print("Generating pairwise data for ranking loss...")
    pairs = []
    for prompt_id in train_df['prompt_id'].unique():
        prompt_df = train_df[train_df['prompt_id'] == prompt_id]
        
        # Create all combinations of 2 answers within the same prompt
        for (idx1, row1), (idx2, row2) in combinations(prompt_df.iterrows(), 2):
            score1, answer1 = row1['score'], row1['answer']
            score2, answer2 = row2['score'], row2['answer']
            prompt_id = int(prompt_id)
            if score1 > score2:
                pairs.append({'prompt_id': prompt_id, 'positive': answer1, 'negative': answer2})
            elif score2 > score1:
                pairs.append({'prompt_id': prompt_id, 'positive': answer2, 'negative': answer1})

    print(f"Generated {len(pairs)} pairs for training.")
    
    # 5. Save Processed Files
    print(f"Saving processed files to {PROCESSED_DATA_DIR}...")
    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    
    with open(PAIRS_FILE, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
            
    print("--- Data Preparation Complete! ---")


if __name__ == "__main__":
    prepare_dataset()