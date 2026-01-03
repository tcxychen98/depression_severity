import torch
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import os

# 1. GPU DETECTION (ROCm)
device_id = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device_id == 0 else "CPU"
print(f"🚀 Using Device: {device_name}")

# 2. CONFIGURATION
INPUT_FILE = "data/Suicide_Detection.csv"
OUTPUT_FILE = "Suicide_Severity_Final.csv"
CHECKPOINT_FILE = "severity_progress.csv"
MODEL_NAME = "valhalla/distilbart-mnli-12-1" 
BATCH_SIZE = 16 

# 3. DATA LOADING & RESUME LOGIC
if not os.path.exists(INPUT_FILE):
    print(f"[INFO] Error: {INPUT_FILE} not found.")
    exit()

df = pd.read_csv(INPUT_FILE)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

suicide_df = df[df['class'] == 'suicide'].copy()
non_suicide_df = df[df['class'] == 'non-suicide'].copy()
non_suicide_df['severity'] = 0

if os.path.exists(CHECKPOINT_FILE):
    processed_df = pd.read_csv(CHECKPOINT_FILE)
    done_count = len(processed_df)
    print(f"[INFO] Resuming from row {done_count}")
    suicide_to_do = suicide_df.iloc[done_count:]
else:
    processed_df = pd.DataFrame()
    suicide_to_do = suicide_df

# 4. INITIALIZE MODEL
print(f"[INFO] Loading {MODEL_NAME} on {device_name}...")

# Optimization: Added torch_dtype=torch.float16 for ROCm speed boost
classifier = pipeline(
    "zero-shot-classification", 
    model=MODEL_NAME, 
    device=device_id,
    torch_dtype=torch.float16 if device_id == 0 else torch.float32
)

labels = [
    "No suicidal risk",                         # 0
    "Passive suicidal ideation",                # 1
    "Active ideation without plan",             # 2
    "Active ideation with method mentioned",    # 3
    "Active ideation with intent and plan",     # 4
    "Immediate crisis or final goodbye"         # 5
]

# 5. STREAMING GENERATOR
def stream_data(texts):
    for text in texts:
        # ROCm handles longer strings better, but 1000 is still safe for performance
        yield str(text)[:1000] 

texts_list = suicide_to_do['text'].tolist()

print("\n" + "="*40)
print(f"[INFO] STARTING PROCESSING ({len(texts_list)} rows)")
print("="*40)

batch_buffer = []
try:
    results_gen = classifier(
        stream_data(texts_list),
        candidate_labels=labels,
        hypothesis_template="Intent {}.",
        batch_size=BATCH_SIZE
    )

    for i, out in enumerate(tqdm(results_gen, total=len(texts_list), desc="Labeling")):
        top_label = out['labels'][0]
        score = labels.index(top_label)
        
        # We index suicide_to_do starting from 'i' 
        row_data = suicide_to_do.iloc[[i]].copy()
        row_data['severity'] = score
        batch_buffer.append(row_data)

        # Save checkpoint every 100 rows
        if (i + 1) % 100 == 0:
            processed_df = pd.concat([processed_df] + batch_buffer)
            processed_df.to_csv(CHECKPOINT_FILE, index=False)
            batch_buffer = []
            # Use tqdm.write to avoid messing up the progress bar
            tqdm.write(f"[INFO] Checkpoint saved at row {len(processed_df)}")

except KeyboardInterrupt:
    print("\n\n [Process Interrupted] Paused by user. Saving current progress...")

# 6. FINAL SAVE
if batch_buffer:
    processed_df = pd.concat([processed_df] + batch_buffer)

# Combine processed suicides with the non-suicide rows
final_all = pd.concat([processed_df, non_suicide_df])
final_all.to_csv(OUTPUT_FILE, index=False)
print(f"\n[INFO] DONE! Total rows processed: {len(final_all)}")
print(f"[INFO] Final file saved: {OUTPUT_FILE}")