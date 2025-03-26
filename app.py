import streamlit as st
from PIL import Image
import tempfile
from inference_sdk import InferenceHTTPClient

# --- Replace with your Roboflow API key and model details ---
ROBOFLOW_API_KEY = "BjCE4IQzwn9VFOGPR9En"  # This is directly used in the InferenceHTTPClient initialization
ROBOFLOW_MODEL_ID = "yeast-cell-counting-v2-ao81v/3"

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

def infer_image(image_file):
    """Sends the image to the Roboflow Inference SDK and returns the results."""
    try:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(image_file.read())
            temp_file_path = tmp_file.name

        # Infer on the local image
        result = CLIENT.infer(temp_file_path, model_id=ROBOFLOW_MODEL_ID)

        # Clean up the temporary file
        import os
        os.remove(temp_file_path)

        return result
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None

def main():
    st.title("Yeast Cell Counting Model Tester")
    st.write("Upload an image to test the yeast cell counting model.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        st.write("### Processing...")

        # Perform inference
        results = infer_image(uploaded_file)

        if results and results.predictions:
            predictions = results.predictions
            num_detections = len(predictions)
            st.write(f"**Detected {num_detections} yeast cell(s):**")

            if num_detections > 0:
                for prediction in predictions:
                    class_name = prediction.class_name
                    confidence = prediction.confidence
                    st.write(f"- **{class_name}:** Confidence: {confidence:.2f}")
            else:
                st.write("No yeast cells detected in this image.")
        elif results:
            st.write("No predictions found in the API response.")

if __name__ == "__main__":
    main()