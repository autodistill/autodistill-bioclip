import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import open_clip
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BioCLIP(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip"
        )
        tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")

        self.model = model.to(DEVICE)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        self.ontology = ontology

    def predict(self, input: Any, confidence: int = 0.5) -> sv.Detections:
        classes = self.ontology.prompts()

        image = load_image(input, return_format="PIL")
        image = self.preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(
                self.tokenizer([f"This is a photo of a {class_}" for class_ in classes])
            )
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            print(np.array(text_probs.cpu().numpy()))

            return sv.Classifications(
                class_id=np.array([i for i in range(len(classes))]),
                confidence=text_probs.cpu().numpy()[0],
            )
