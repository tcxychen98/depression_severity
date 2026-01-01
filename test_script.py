import torch
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import time
import os

# 1. SETUP DEVICE (Apple Silicon GPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using GPU (MPS)")
else:
    device = torch.device("cpu")
    print("GPU not found, using CPU (Slow!)")

# 2. LOAD DATA
file_path = 'data/Suicide_Detection.csv'
checkpoint_path = 'severity_checkpoints.csv'

df = pd.read_csv(file_path)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Prepare the dataframes
suicide_df = df[df['class'] == 'suicide'].copy()
non_suicide_df = df[df['class'] == 'non-suicide'].copy()
non_suicide_df['severity'] = 0

# 3. RESUME LOGIC (If you have to restart)
if os.path.exists(checkpoint_path):
    processed_df = pd.read_csv(checkpoint_path)
    processed_ids = set(processed_df['text'].tolist()) # Using text as ID for simplicity
    print(f"Resuming: {len(processed_df)} rows already labeled.")
    # Filter out what we already did
    suicide_to_process = suicide_df[~suicide_df['text'].isin(processed_ids)]
else:
    processed_df = pd.DataFrame()
    suicide_to_process = suicide_df

# 4. INITIALIZE MODEL
# Using a slightly faster model that works well on Mac
model_name = "facebook/bart-large-mnli" 
classifier = pipeline("zero-shot-classification", 
                      model=model_name, 
                      device="mps") # Explicitly set to mps

labels = [
    "No suicidal risk",                         # 0
    "Passive suicidal ideation",                # 1
    "Active ideation without plan",             # 2
    "Active ideation with method mentioned",    # 3
    "Active ideation with intent and plan",     # 4
    "Immediate crisis or final goodbye"         # 5
]

# 5. GENERATOR FOR STREAMING
def stream_data(texts):
    for text in texts:
        yield str(text)[:1000] # Truncate for speed

texts_list = suicide_to_process['text'].tolist()
total_to_do = len(texts_list)

print(f"\nTotal left to process: {total_to_do}")
print("="*40)

# 6. PROCESSING LOOP
batch_results = []
start_time = time.time()

try:
    # Use a small batch size for Mac stability
    results_gen = classifier(
        stream_data(texts_list),
        candidate_labels=labels,
        hypothesis_template="This person is expressing {}.",
        batch_size=4 
    )

    for i, out in enumerate(tqdm(results_gen, total=total_to_do, desc="MPS Labeling")):
        top_label = out['labels'][0]
        score = labels.index(top_label)
        
        # Create a tiny dataframe for this one row
        current_row = suicide_to_process.iloc[[i]].copy()
        current_row['severity'] = score
        batch_results.append(current_row)

        # Every iteration "Alive" feedback
        if i % 1 == 0:
            print(f"\r[ALIVE] Row {i+1}/{total_to_do} | Score: {score}", end="")

        # Save Checkpoint every 100 rows
        if (i + 1) % 100 == 0:
            checkpoint_df = pd.concat([processed_df] + batch_results)
            checkpoint_df.to_csv(checkpoint_path, index=False)
            # We don't want to re-save the whole thing every time, 
            # so we keep processed_df growing
            processed_df = checkpoint_df
            batch_results = [] 

except KeyboardInterrupt:
    print("\nPaused by user. Progress saved.")

# 7. FINAL MERGE
final_suicide = pd.concat([processed_df] + batch_results)
final_all = pd.concat([final_suicide, non_suicide_df])
final_all.to_csv("Final_Labeled_Suicide_Data.csv", index=False)
print("\n\nMISSION COMPLETE: Data saved to Final_Labeled_Suicide_Data.csv")