from ultralytics import YOLO

def main():
    # Load a pre-trained YOLO model (recommended for training)
    model = YOLO("yolo26s.pt") 

    # Train the model using the dataset config
    results = model.train(
        data="dataset.yaml",   # Path to the dataset config we created
        epochs=40,             # Number of training epochs
        imgsz=640,             # Target image size
        batch=8,              # Batch size
        device="cpu",          # Change to "0" if you have a CUDA compatible GPU
        project="runs3",  # Where to save the results
        name="detrac_yolo"     # Name of the training run
    )

if __name__ == "__main__":
    main()
    