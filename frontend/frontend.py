import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.model_service import ModelService

# Session state initialization
if "diagnosed" not in st.session_state:
    st.session_state.diagnosed = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "model_service" not in st.session_state:
    st.session_state.model_service = ModelService()


@st.cache_resource
def initialize_model_service():
    """Initialize and load model (cached by Streamlit)"""
    service = ModelService()
    model, labels_map, idx_to_class = service.load_model()

    if model is not None:
        service.model = model
        service.labels_map = labels_map
        service.idx_to_class = idx_to_class

    return service


def run():
    """Main Streamlit app"""
    st.set_page_config(page_title="Leaf Disease Recognition", layout="wide")

    st.title("Mango Leaf Disease Recognition")

    with st.container():
        st.write(
            "AI-powered application using SqueezeNet CNN to diagnose mango leaf diseases"
        )
        st.markdown("---")

        # Load model and transforms at startup
        with st.spinner("Loading model..."):
            model_service = initialize_model_service()

            if model_service.model is not None:
                st.session_state.model_service = model_service
                st.success("✓ Model loaded successfully")
            else:
                st.error("Failed to load model. Check model path in config.yaml")

    with st.container():
        st.subheader("Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose a mango leaf image",
            type=["png", "jpg", "jpeg", "webp"],
            key="leaf_uploader",
        )

        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
        elif (
            uploaded_file is None and st.session_state.get("uploaded_file") is not None
        ):
            st.session_state.uploaded_file = None
            st.session_state.diagnosed = False
            st.warning("File cleared. Upload a new image to diagnose.")

        def handle_diagnose():
            if (
                st.session_state.uploaded_file is not None
                and st.session_state.model_service.model is not None
            ):
                st.session_state.diagnosed = True
            elif st.session_state.model_service.model is None:
                st.error("Model not loaded. Check configuration.")
            else:
                st.warning("Please upload an image first!")

        col_button = st.columns([0.2, 0.6, 0.2])[1]
        with col_button:
            st.button(
                "Diagnose",
                on_click=handle_diagnose,
                type="primary",
                use_container_width=True,
            )

        st.markdown("---")
    if (
        st.session_state.get("diagnosed", False)
        and st.session_state.get("uploaded_file") is not None
        and st.session_state.get("model_service") is not None
        and st.session_state.model_service.model is not None
    ):
        uploaded_file = st.session_state.uploaded_file
        model_service = st.session_state.model_service

        try:
            # Load and prepare image
            image_data = io.BytesIO(uploaded_file.getvalue())
            image_pil = Image.open(image_data).convert("RGB")

            # Run prediction
            result = model_service.predict(image_pil)
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            top_3 = result["top_3"]

            # Display results
            st.subheader(f"Diagnosis Result")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(image_pil, caption="Uploaded Image", width="stretch")

            with col2:
                st.metric(
                    label="Predicted Disease",
                    value=predicted_class.replace("_", " "),
                    delta=f"Confidence: {confidence*100:.1f}%",
                )

                if confidence < 0.6:
                    st.warning("Low confidence - consider consulting an expert")
                elif confidence >= 0.8:
                    st.success("High confidence prediction")
                else:
                    st.info("Moderate confidence")

            st.markdown("---")

            # Display disease information
            disease_info = model_service.get_disease_info(predicted_class)
            if disease_info:
                col_info, col_rec = st.columns(2)

                with col_info:
                    with st.expander(f"Information", expanded=True):
                        st.write(disease_info["info"])

                with col_rec:
                    with st.expander("Recommended Treatment", expanded=True):
                        st.write(disease_info["recommendations"])
            else:
                st.warning(f"No disease information found for '{predicted_class}'")

            st.markdown("---")

            # Show top-3 predictions
            st.subheader("All Predictions (Top-3)")

            predictions_df_data = []
            for rank, (class_name, prob) in enumerate(top_3, 1):
                predictions_df_data.append(
                    {
                        "Rank": rank,
                        "Disease": class_name.replace("_", " "),
                        "Probability": f"{prob*100:.2f}%",
                    }
                )

            st.dataframe(predictions_df_data, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error during diagnosis: {e}")
            st.error("Please ensure the image is a valid leaf photo.")

    elif st.session_state.get("diagnosed", False) and (
        st.session_state.get("uploaded_file") is None
        or st.session_state.get("model_service") is None
        or st.session_state.model_service.model is None
    ):
        st.warning("Unable to process diagnosis. Please check:")
        if (
            st.session_state.get("model_service") is not None
            and st.session_state.model_service.model is None
        ):
            st.write("- Model weights file path in config.yaml")
        if st.session_state.get("uploaded_file") is None:
            st.write("- Upload an image file")

    elif not st.session_state.get("diagnosed", False):
        st.info("Upload a mango leaf image and click 'Diagnose' to get started")


if __name__ == "__main__":
    run()
