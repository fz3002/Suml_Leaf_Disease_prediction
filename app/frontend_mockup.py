import io
import os
import streamlit as st
from PIL import Image

if 'diagnosed' not in st.session_state:
    st.session_state.diagnosed = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

MOCK_FILE_PATH = os.path.join(
    "app", "data", "placeholders", "mango_leaf_placeholder.jpg")
MOCK_DISEASE = "Anthracnose (Mock Diagnosis)"
MOCK_INFO = "Anthracnose is a common fungal disease of mangoes, especially in humid conditions."
MOCK_RECOMMENDATIONS = "Apply copper-based fungicides regularly, particularly during flowering and fruiting periods."

st.set_page_config(page_title="Leaf Disease Recognition", layout="centered")

st.title("Leaf Disease Recognition")

with st.container():
    st.write(
        "Application uses image recognition to diagnose mango leaf diseases")
    st.markdown("---")

with st.container():
    uploaded_file = st.file_uploader("Upload photo of mango leaf to diagnose",
                                     type=["png", "jpg", "jpeg", "webp", "gif"]
                                     )

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    def handle_diagnose():
        if st.session_state.uploaded_file is not None:
            st.session_state.diagnosed = True
        else:
            st.warning("File not uploaded!!!")

    st.button("Diagnose", on_click=handle_diagnose, type="primary")
    st.markdown("---")
if st.session_state.diagnosed and st.session_state.uploaded_file is not None:
    st.subheader(f"Diagnosis: {MOCK_DISEASE}")

    with st.container():
        col1, col2 = st.columns([1, 1])

        with col1:
            if uploaded_file is not None:
                try:
                    image_data = io.BytesIO(uploaded_file.getvalue())
                    image_to_display = Image.open(image_data)
                    st.image(image_to_display, caption="Uploaded Photo")
                except Exception as e:
                    st.error(f"Error during file upload: {e}")

        with col2:
            try:
                st.image(MOCK_FILE_PATH, caption="Heatmap Photo")
            except FileNotFoundError:
                st.error(
                    f"Error: File not found: {MOCK_FILE_PATH}")

    st.markdown('---')

    with st.container():
        st.markdown('### Information and Recommendations')

        with st.expander(f"Information about disease {MOCK_DISEASE}"):
            st.write(MOCK_INFO)

        with st.expander("Recommendations of treatment"):
            st.write(MOCK_RECOMMENDATIONS)

elif st.session_state.diagnosed and st.session_state.uploaded_file is None:
    st.warning("File to diagnose not available")


elif st.session_state.diagnosed is False:
    st.info("Waiting for file to diagnose")
