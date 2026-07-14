from ultralytics import YOLO
import cv2
import os
import time


class Detector:
    """
    Construction Safety Detector

    Supports:
    1. Image Detection
    2. Video Detection
    3. Webcam Detection
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5
    ):
        """
        Load YOLO model only once.
        """

        self.model = YOLO(model_path)
        self.confidence = confidence

    # ==========================================================
    # PRIVATE METHODS
    # ==========================================================

    def _create_detection_summary(self, results):
        """
        Create Detection Summary
        """

        detections = []

        boxes = results[0].boxes

        for box in boxes:

            class_id = int(box.cls[0])

            class_name = results[0].names[class_id]

            confidence = float(box.conf[0])

            detections.append({

                "Object": class_name,

                "Confidence (%)": round(
                    confidence * 100,
                    2
                )

            })

        return detections

    def _average_confidence(self, detections):

        if len(detections) == 0:
            return 0

        return round(

            sum(
                d["Confidence (%)"]
                for d in detections
            ) / len(detections),

            2

        )

    # ==========================================================
    # IMAGE DETECTION
    # ==========================================================

    def detect_image(
        self,
        image_path
    ):

        start_time = time.time()

        results = self.model.predict(

            source=image_path,

            conf=self.confidence,

            save=True,

            project="outputs",

            name="image",

            exist_ok=True

        )

        inference_time = round(

            time.time() - start_time,

            3

        )

        save_dir = results[0].save_dir

        output_path = os.path.join(

            save_dir,

            os.path.basename(image_path)

        )

        detections = self._create_detection_summary(
            results
        )

        total_objects = len(detections)

        average_confidence = self._average_confidence(
            detections
        )

        return {

            "output_path": output_path,

            "detections": detections,

            "total_objects": total_objects,

            "average_confidence": average_confidence,

            "inference_time": inference_time

        }

    # ==========================================================
    # VIDEO DETECTION
    # ==========================================================

    def detect_video(self, video_path):

        start_time = time.time()

        results = self.model.predict(
        source=video_path,
        conf=self.confidence,
        save=True,
        project="outputs",
        name="video",
        exist_ok=True
    )

        inference_time = round(time.time() - start_time, 3)

        save_dir = results[0].save_dir

        video_files = [
        f for f in os.listdir(save_dir)
        if f.lower().endswith((".avi", ".mp4", ".mov", ".mkv"))
    ]

        if not video_files:
            raise FileNotFoundError("Processed video not found.")

    # Latest processed video
        video_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(save_dir, f)),
        reverse=True
    )

        output_video = os.path.join(save_dir, video_files[0])

        print("Output Video:", output_video)

        if output_video.lower().endswith(".avi"):

            output_video = self.convert_avi_to_mp4(output_video)

        return {
        "output_video": output_video,
        "inference_time": inference_time
    }

    # ==========================================================
    # WEBCAM DETECTION
    # ==========================================================

    def detect_webcam(self):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            print("Error : Webcam Not Found")

            return

        while True:

            ret, frame = cap.read()

            if not ret:

                break

            results = self.model.predict(

                source=frame,

                conf=self.confidence,

                verbose=False

            )

            annotated_frame = results[0].plot()

            cv2.imshow(

                "Construction Safety Detector",

                annotated_frame

            )

            key = cv2.waitKey(1)

            if key == ord("q"):

                break

        cap.release()

        cv2.destroyAllWindows()


    # ==========================================================
    # CHANGE CONFIDENCE
    # ==========================================================

    def set_confidence(
        self,
        confidence
    ):

        self.confidence = confidence

    def convert_avi_to_mp4(self, avi_path):

        mp4_path = avi_path.replace(".avi", ".mp4")

        cap = cv2.VideoCapture(avi_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
        mp4_path,
        fourcc,
        fps,
        (width, height)
    )

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            out.write(frame)

        cap.release()

        out.release()

        return mp4_path

    # ==========================================================
    # MODEL INFORMATION
    # ==========================================================

    def model_info(self):

        return {

            "Model Path": self.model.ckpt_path,

            "Confidence": self.confidence

        }
        