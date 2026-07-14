import streamlit as st
import os
from detector import Detector

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Construction Safety Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main-title{
    font-size:40px;
    font-weight:bold;
    color:#2E86C1;
}

.metric-box{
    padding:15px;
    border-radius:10px;
    background:#f2f2f2;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_detector(confidence):
    detector = Detector("models/best.pt")
    detector.set_confidence(confidence)
    return detector

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🦺 Construction Safety")

st.sidebar.markdown("---")

mode = st.sidebar.radio(

    "Choose Detection Mode",

    (

        "🖼 Image Detection",

        "🎥 Video Detection",

        "📷 Webcam Detection"

    )

)

confidence = st.sidebar.slider(

    "Confidence Threshold",

    min_value=0.10,

    max_value=1.00,

    value=0.50,

    step=0.05

)

detector = load_detector(confidence)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Model : YOLO11

Framework : Ultralytics

Developer : Jahid
"""
)

# ==========================================================
# IMAGE DETECTION
# ==========================================================

if mode=="🖼 Image Detection":

    st.markdown(
        "<p class='main-title'>🖼 Image Detection</p>",
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader(

        "Upload Image",

        type=["jpg","jpeg","png"]

    )

    if uploaded_image:

        col1,col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(

                uploaded_image,

                width="stretch"

            )

        if st.button("🔍 Detect Objects"):

            temp_path = os.path.join(

                "temp",

                uploaded_image.name

            )

            os.makedirs("temp",exist_ok=True)

            with open(temp_path,"wb") as f:

                f.write(uploaded_image.getbuffer())

            with st.spinner("Running YOLO Detection..."):

                result = detector.detect_image(temp_path)

            with col2:

                st.subheader("Detected Image")

                st.image(

                    result["output_path"],

                    width="stretch"

                )

            st.success("Detection Completed")

            st.divider()

            st.subheader("📊 Detection Summary")

            st.table(result["detections"])

            c1,c2,c3 = st.columns(3)

            with c1:

                st.metric(

                    "Objects",

                    result["total_objects"]

                )

            with c2:

                st.metric(

                    "Average Confidence",

                    f'{result["average_confidence"]}%'

                )

            with c3:

                st.metric(

                    "Inference Time",

                    f'{result["inference_time"]} sec'

                )

# ==========================================================
# VIDEO DETECTION
# ==========================================================

elif mode == "🎥 Video Detection":

    st.markdown(
        "<p class='main-title'>🎥 Video Detection</p>",
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video:

        st.subheader("Original Video")

        st.video(uploaded_video)

        if st.button("🎬 Detect Video"):

            os.makedirs("temp", exist_ok=True)

            temp_video = os.path.join(
                "temp",
                uploaded_video.name
            )

            with open(temp_video, "wb") as f:

                f.write(uploaded_video.getbuffer())

            with st.spinner("Processing Video..."):

                result = detector.detect_video(temp_video)

            st.success("Video Detection Completed ✅")
            st.write("Output Path:", result["output_video"])
            st.write("File Exists:", os.path.exists(result["output_video"]))

            st.subheader("Processed Video")
            # st.video(result["output_video"])
            with open(result["output_video"], "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes,format="video/mp4")
            
            st.metric(

                "Inference Time",

                f'{result["inference_time"]} sec'

            )

            with open(result["output_video"], "rb") as file:

                st.download_button(

                    label="⬇ Download Processed Video",

                    data=file,

                    file_name=os.path.basename(
                        result["output_video"]
                    ),

                    mime="video/x-msvideo"

                )


# ==========================================================
# WEBCAM DETECTION
# ==========================================================

elif mode == "📷 Webcam Detection":

    st.markdown(
        "<p class='main-title'>📷 Live Webcam Detection</p>",
        unsafe_allow_html=True
    )

    st.info(
        """
        Instructions

        1. Click Start Webcam

        2. Press Q to Exit Webcam

        3. Close OpenCV Window
        """
    )

    if st.button("▶ Start Webcam"):

        detector.detect_webcam()

        st.success("Webcam Closed Successfully")


# ==========================================================
# SIDEBAR INFORMATION
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.subheader("Project Features")

st.sidebar.success("✔ Image Detection")

st.sidebar.success("✔ Video Detection")

st.sidebar.success("✔ Webcam Detection")

st.sidebar.success("✔ Detection Summary")

st.sidebar.success("✔ Object Counter")

st.sidebar.success("✔ Average Confidence")

st.sidebar.success("✔ Inference Time")

st.sidebar.success("✔ Download Result")


st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

info = detector.model_info()

st.sidebar.write("**Model:**")

st.sidebar.write(info["Model Path"])

st.sidebar.write("**Confidence:**")

st.sidebar.write(info["Confidence"])

# ==========================================================
# DOWNLOAD IMAGE (IMAGE MODE)
# ==========================================================

if mode == "🖼 Image Detection":

    try:

        if uploaded_image is not None:

            if 'result' in locals():

                with open(result["output_path"], "rb") as file:

                    st.download_button(

                        label="⬇ Download Detected Image",

                        data=file,

                        file_name=os.path.basename(
                            result["output_path"]
                        ),

                        mime="image/jpeg"

                    )

    except Exception:

        pass


# ==========================================================
# TEMP CLEANUP
# ==========================================================

try:

    if os.path.exists("temp"):

        for file in os.listdir("temp"):

            path = os.path.join("temp", file)

            try:

                os.remove(path)

            except:

                pass

except:

    pass


# ==========================================================
# ABOUT PROJECT
# ==========================================================

st.sidebar.markdown("---")

with st.sidebar.expander("ℹ About Project"):

    st.write("""
### Construction Safety Detection System

This application is built using

- YOLO11
- Ultralytics
- OpenCV
- Streamlit
- Python

Features

✔ Image Detection

✔ Video Detection

✔ Webcam Detection

✔ PPE Detection

✔ Download Results

✔ Detection Summary
""")


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;'>

<h4>
🦺 Construction Safety Detection System
</h4>

Built using
YOLO11 • Streamlit • OpenCV • Python

</div>
""",
unsafe_allow_html=True
)