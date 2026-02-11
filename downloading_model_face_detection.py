from huggingface_hub import hf_hub_download

# Download just the .pt file since the model was too big to push to the github repo
model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection",
    filename="model.pt",
    local_dir="models/yolo_face"  # Optional folder
)

print(f"YOLO face model saved to: {model_path}")
