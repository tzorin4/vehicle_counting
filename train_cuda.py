from ultralytics import YOLO
import torch

def main():
    print("cuda available:", torch.cuda.is_available())
    # Load a pre-trained YOLO model
    model = YOLO("yolo26s.pt")

    # Train the model using the dataset config
    results = model.train(
        data="cuda_dataset.yaml",  # full path to your YAML on Drive
        epochs=50,
        imgsz=640,
        batch=24,               # reduced batch size for Colab GPU (adjust if you OOM)
        device=0,               # use CUDA GPU 0 (int) or "cuda:0"
        project="runs",
        name="detrac_yolo"
    )

if __name__ == "__main__":
    main()