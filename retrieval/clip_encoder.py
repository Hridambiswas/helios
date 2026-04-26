# retrieval/clip_encoder.py — Multi-modal CLIP encoder (text + image)
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger("helios.retrieval.clip_encoder")

_MODEL_NAME = "openai/clip-vit-base-patch32"
_model: CLIPModel | None = None
_processor: CLIPProcessor | None = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _load() -> tuple[CLIPModel, CLIPProcessor]:
    global _model, _processor
    if _model is None:
        logger.info("Loading CLIP model %s on %s ...", _MODEL_NAME, _device)
        _model = CLIPModel.from_pretrained(_MODEL_NAME).to(_device).eval()
        _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        logger.info("CLIP model ready")
    return _model, _processor


@torch.inference_mode()
def encode_text(text: str | list[str]) -> list[list[float]]:
    """
    Encode one or more text strings into CLIP text embeddings.
    Returns a list of float lists (one per input).
    """
    model, processor = _load()
    texts = [text] if isinstance(text, str) else text
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(_device)
    features = model.get_text_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)   # L2-normalise
    return features.cpu().numpy().tolist()


@torch.inference_mode()
def encode_image(image: Union[str, Path, Image.Image, bytes]) -> list[float]:
    """
    Encode a single image into a CLIP image embedding.
    Accepts a file path, PIL Image, or raw bytes.
    """
    model, processor = _load()

    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    elif isinstance(image, bytes):
        import io
        pil = Image.open(io.BytesIO(image)).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    inputs = processor(images=pil, return_tensors="pt").to(_device)
    features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].tolist()


@torch.inference_mode()
def encode_images_batch(images: list[Union[str, Path, Image.Image, bytes]]) -> list[list[float]]:
    """Batch-encode multiple images — more efficient than looping encode_image."""
    model, processor = _load()
    pils = []
    for img in images:
        if isinstance(img, (str, Path)):
            pils.append(Image.open(img).convert("RGB"))
        elif isinstance(img, bytes):
            import io
            pils.append(Image.open(io.BytesIO(img)).convert("RGB"))
        else:
            pils.append(img.convert("RGB"))

    inputs = processor(images=pils, return_tensors="pt").to(_device)
    features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()


def embedding_dim() -> int:
    model, _ = _load()
    return model.config.projection_dim
