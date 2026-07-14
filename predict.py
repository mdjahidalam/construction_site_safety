from ultralytics import YOLO
import os

# Load model once
model = YOLO("models/best.pt")


def detect(image_path):

    results = model.predict(
        source=image_path,
        conf=0.5,
        save=True,
        project="outputs",
        name="predict",
        exist_ok=True
    )

    # Actual folder where YOLO saved image
    save_dir = results[0].save_dir

    output_path = os.path.join(
        save_dir,
        os.path.basename(image_path)
    )

    print("Save Directory :", save_dir)
    print("Output Path :", output_path)

    return output_path, results