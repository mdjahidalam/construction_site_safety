from ultralytics import YOLO

# Load trained model
model = YOLO("models/best.pt")


def detect_video(video_path):

    results = model.predict(
        source=video_path,
        conf=0.5,
        save=True,
        project="outputs",
        name="video",
        exist_ok=True
    )

    return results[0].save_dir