from ultralytics import YOLO

def main():
    # Load a pre-trained YOLO model (recommended for training)
    model = YOLO("yolov8n.pt") 

    # Train the model using the dataset config
    results = model.train(
        data="dataset.yaml",   # Path to the dataset config we created
        epochs=50,             # Number of training epochs
        imgsz=640,             # Target image size
        batch=50,              # Batch size
        device="0",          # Change to "0" if you have a CUDA compatible GPU
        project="runs/train",  # Where to save the results
        name="detrac_yolo"     # Name of the training run
    )

if __name__ == "__main__":
    main()
    