from ultralytics import YOLO
import torch

def main():
    print("cuda available:", torch.cuda.is_available())

    model = YOLO("yolo26s.pt")

    results = model.train(
        data="cuda_dataset.yaml",  
        epochs=50,
        imgsz=640,
        batch=24,               
        device=0,              
        project="runs",
        name="detrac_yolo"
    )

if __name__ == "__main__":
    main()