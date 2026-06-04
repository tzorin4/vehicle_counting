from ultralytics import YOLO

def main():
    # Load the best model weights from your training run
    # If the path is different, update it accordingly.
    model_path = "runs/detect/runs/train/detrac_yolo-2/weights/best.pt"
    
    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Make sure training finished successfully.")
        return

    # 1. EVALUATION (Validation)
    # This evaluates the model's metrics (mAP, Precision, Recall) on your validation dataset
    print("--- Running Evaluation on Validation Set ---")
    metrics = model.val(data="dataset.yaml")
    
    print("\nMetrics summary:")
    print(f"Mean Average Precision (mAP50-95): {metrics.box.map:.4f}")
    
    # 2. INFERENCE (Testing on unseen images)
    # Give it a path to a test image or folder of images to get visual predictions
    # Example: you can pass the path to the validation images folder
    print("\n--- Running Predictions on Sample Images ---")
    results = model.predict(
        source="archive/content/UA-DETRAC/DETRAC_Upload/images/val", # Directory or specific image path
        save=True,          # Save images with drawn bounding boxes
        conf=0.25,          # Confidence threshold (only show detections > 25% confidence)
        project="runs/test",# Where to save the results
        name="detrac_test"
    )
    
    print("Testing complete. Check the 'runs/test/detrac_test' folder for the images with bounding boxes!")

if __name__ == "__main__":
    main()
