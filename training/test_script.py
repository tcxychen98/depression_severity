import torch
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import os
import time

# 1. UNIVERSAL DEVICE DETECTION
def get_device():
    # Check for Mac GPU
    if torch.backends.mps.is_available():
        print("🚀 Device: Mac GPU (MPS) detected.")
        return "mps"
    
    # Check for Windows AMD GPU (DirectML)
    try:
        import torch_directml
        print("🚀 Device: Windows AMD GPU (DirectML) detected.")
        return torch_directml.device()
    except ImportError:
        pass
    
    print("⚠️ Device: GPU not found. Falling back to CPU.")
    return "cpu"

device = get_device()

# 2. CONFIGURATION
INPUT_FILE = "data/Suicide_Detection.csv"
OUTPUT_FILE = "Suicide_Severity_Final.csv"
CHECKPOINT_FILE = "severity_progress.csv"
MODEL_NAME = "valhalla/distilbart-mnli-12-1" # Smaller, faster model

# Set batch size based on platform
# Windows DirectML often prefers smaller batches (4-8), Mac can handle 12+
BATCH_SIZE = 8 

# 3. DATA LOADING & RESUME LOGIC
df = pd.read_csv(INPUT_FILE)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

suicide_df = df[df['class'] == 'suicide'].copy()
non_suicide_df = df[df['class'] == 'non-suicide'].copy()
non_suicide_df['severity'] = 0

if os.path.exists(CHECKPOINT_FILE):
    processed_df = pd.read_csv(CHECKPOINT_FILE)
    done_count = len(processed_df)
    print(f"🔄 Resuming from row {done_count}")
    suicide_to_do = suicide_df.iloc[done_count:]
else:
    processed_df = pd.DataFrame()
    suicide_to_do = suicide_df

# 4. INITIALIZE MODEL
print(f"📦 Loading {MODEL_NAME}...")

# Note: For DirectML, we sometimes need to load the model manually 
# if the pipeline 'device' argument acts up.
classifier = pipeline("zero-shot-classification", 
                      model=MODEL_NAME, 
                      device=device)

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
        yield str(text)[:1000] # Truncate for performance

texts_list = suicide_to_do['text'].tolist()

print("\n" + "="*40)
print(f"STARTING PROCESSING ({len(texts_list)} rows)")
print("="*40)

batch_buffer = []
try:
    results_gen = classifier(
        stream_data(texts_list),
        candidate_labels=labels,
        hypothesis_template="This person is expressing {}.",
        batch_size=BATCH_SIZE
    )

    for i, out in enumerate(tqdm(results_gen, total=len(texts_list), desc="Labeling")):
        top_label = out['labels'][0]
        score = labels.index(top_label)
        
        row_data = suicide_to_do.iloc[[i]].copy()
        row_data['severity'] = score
        batch_buffer.append(row_data)

        # Immediate feedback
        print(f"\r[ALIVE] Last Score: {score} | Row: {i+1}", end="")

        # Save checkpoint every 100 rows
        if (i + 1) % 100 == 0:
            processed_df = pd.concat([processed_df] + batch_buffer)
            processed_df.to_csv(CHECKPOINT_FILE, index=False)
            batch_buffer = []

except KeyboardInterrupt:
    print("\n\n🛑 Paused by user.")

# 6. FINAL SAVE
if batch_buffer:
    processed_df = pd.concat([processed_df] + batch_buffer)

final_all = pd.concat([processed_df, non_suicide_df])
final_all.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ DONE! File saved: {OUTPUT_FILE}")