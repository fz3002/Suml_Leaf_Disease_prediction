"""
Model Service - handles model loading, inference, and disease database
"""

import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st

from src.model.model import SqueezeNet
from src.data_preparation.preprocessing.data_util import (
    load_config_file,
    load_labels_map,
)
from src.data_preparation.data_handler.transform import Transforms

# Load disease database from JSON
_DB_PATH = Path(__file__).parent / "data" / "disease_database.json"
with open(_DB_PATH, "r", encoding="utf-8") as f:
    DISEASE_DATABASE = json.load(f)


class ModelService:
    """Handles model loading, inference, and disease information"""

    def __init__(self):
        self.model = None
        self.labels_map = None
        self.idx_to_class = None
        self.transforms = None
        self.device = None
        self.config = None

    @st.cache_resource
    def load_model_cached(self):
        """Load pre-trained SqueezeNet model with weights (cached by Streamlit)"""
        return self.load_model()

    def load_model(self):
        """Load pre-trained SqueezeNet model with weights (from notebook 7 logic)."""
        try:
            # Load config.yaml file
            config = load_config_file()
            self.config = config

            # Prepare model from config.yaml file
            device_str = config["train"]["device"]  
            self.device = torch.device(device_str)  
            num_classes = config["model"]["num_classes"]
            model = SqueezeNet(num_classes=num_classes).to(
                self.device
            )

            # Load weights (build absolute path from project_path + weights)
            project_path = config["path"]["project_path"]
            weights_rel = config["weights"]
            import os

            weights_path = os.path.join(project_path, weights_rel)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            # Load labels map
            self.labels_map = load_labels_map()
            self.idx_to_class = {v: k for k, v in self.labels_map.items()}
            self.transforms = Transforms()
            self.model = model

            return (model, self.labels_map, self.idx_to_class)
        except Exception as e:
            import traceback

            error_msg = f"Error loading model: {e}\n{traceback.format_exc()}"
            st.error(error_msg)
            print(error_msg)
            return None, None, None

    def get_transforms(self):
        """Get validation transforms (no augmentation)."""
        if self.transforms is None:
            self.transforms = Transforms()
        return self.transforms

    def predict(self, image_pil: Image.Image) -> dict:
        """
        Predict disease from image.

        Args:
            image_pil: PIL Image object (RGB)

        Returns:
            dict with keys:
                - predicted_class: str
                - confidence: float
                - probabilities: dict
                - top_3: list of tuples
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            image = image_pil.convert("RGB")

            # Preparing Transforms
            test_transform = self.get_transforms().get_test_transform()

            # Transforming image
            image = test_transform(image)
            image = image.unsqueeze(
                dim=0
            )
            image = image.to(self.device)  # Putting image on device (GPU or CPU)

            with torch.no_grad():
                output = self.model(image)  # inference
                probabilities = F.softmax(output, dim=1)  # Turn logits to probabilities

            # Get predicted class and confidence
            class_idx = output.argmax(dim=1).item()
            confidence = probabilities[0, class_idx].item()
            predicted_class = self.idx_to_class.get(class_idx, "Unknown")

            # Get probabilities for each class (sorted)
            probs_flat = probabilities.squeeze()
            all_probs = {
                self.idx_to_class[i]: p.item() * 100 for i, p in enumerate(probs_flat)
            }

            # Get top-3 predictions
            top3_probs, top3_indices = torch.topk(
                probabilities[0], k=min(3, len(self.idx_to_class))
            )
            top_3 = [
                (self.idx_to_class.get(idx.item(), "Unknown"), prob.item())
                for prob, idx in zip(top3_probs, top3_indices)
            ]

            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": all_probs,
                "top_3": top_3,
            }
        except Exception as e:
            raise Exception(f"Prediction error: {e}")

    def get_disease_info(self, disease_name: str) -> dict or None:
        """Get disease information and recommendations"""
        # Normalize disease_name: replace underscores with spaces to match database keys
        normalized_name = disease_name.replace("_", " ")

        if normalized_name in DISEASE_DATABASE:
            return DISEASE_DATABASE[normalized_name]

        if disease_name in DISEASE_DATABASE:
            return DISEASE_DATABASE[disease_name]

        print(
            f"[WARNING] Disease '{disease_name}' (normalized: '{normalized_name}') not found in database. Available: {list(DISEASE_DATABASE.keys())}"
        )
        return None

    def get_all_diseases(self) -> list:
        """Get list of all supported diseases"""
        return list(DISEASE_DATABASE.keys())

    def get_device_info(self) -> str:
        """Get device information (GPU or CPU)"""
        return str(self.device) if self.device else "Not loaded"
