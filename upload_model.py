from huggingface_hub import HfApi, login

# 1. Login (paste your token when prompted)
login()

# 2. Upload the folder
api = HfApi()
api.upload_folder(
    folder_path="./final_severity_model",
    repo_id="tcxy98/suicide-severity-deberta",
    repo_type="model"
)