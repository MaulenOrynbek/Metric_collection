from huggingface_hub import snapshot_download

# Download the full model folder locally since the model was too big to push to the github repo
snapshot_download(
    repo_id="DrGM/DrGM-ConvNeXt-V2L-Facial-Emotion-Recognition",
    local_dir="models/emotion_convnext",  # Your desired folder
    local_dir_use_symlinks=False         # Copies files fully (safer for offline)
)

print("Emotion model downloaded to ./models/emotion_convnext")

