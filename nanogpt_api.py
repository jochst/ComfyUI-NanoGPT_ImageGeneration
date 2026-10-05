"""Shared NanoGPT API helpers: auth, model discovery, tensor <-> image conversion."""
import base64
import io
import json
import os

import numpy as np
import requests
import torch
from PIL import Image, ImageOps

BASE_URL = "https://nano-gpt.com"
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(NODE_DIR, "models_snapshot.json")
KEY_FILE = os.path.join(NODE_DIR, "api_key.txt")

# Fallback default for the prompt node when the live text-model list can't be fetched.
DEFAULT_TEXT_MODEL = "z-ai/glm-5.3-uncensored"


def get_api_key():
    """API key from NANOGPT_API_KEY env var, else api_key.txt next to this file."""
    key = os.environ.get("NANOGPT_API_KEY", "").strip()
    if not key and os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            # Accept a bare key or a "NANOGPT_API_KEY=..." line.
            key = f.read().strip().split("=", 1)[-1].strip().strip('"').strip("'")
    if not key:
        raise Exception(
            "NanoGPT API key missing. Set NANOGPT_API_KEY or put the key in "
            f"{KEY_FILE}"
        )
    return key


def headers():
    return {"Authorization": "Bearer " + get_api_key(), "Content-Type": "application/json"}


def raise_for_status(r):
    if r.status_code >= 400:
        raise Exception(f"NanoGPT HTTP {r.status_code}: {r.text[:1000]}")


# ---------------------------------------------------------------- model lists

def is_nsfw(model):
    """Mirror the website's 'nsfw' filter: nsfw tag, or nsfw-capable '-spicy' variant."""
    tags = model.get("tags") or []
    caps = model.get("capabilities") or {}
    return "nsfw" in tags or ("spicy" in model.get("id", "") and caps.get("nsfw"))


def is_raster(model):
    """Exclude vector (SVG) models; ComfyUI IMAGE needs raster output."""
    return "vector" not in model.get("id", "") and "svg" not in (model.get("tags") or [])


def is_paid(model):
    """Paid = pay-per-use pricing present (not subscription-included)."""
    p = model.get("pricing") or {}
    return any(k for k in p if k not in ("currency", "note"))


def _fetch(path):
    r = requests.get(BASE_URL + path, timeout=10)
    r.raise_for_status()
    return r.json().get("data", [])


def _load_media_models():
    """Live nsfw+paid image/video models, falling back to the bundled snapshot."""
    try:
        live = {"image": _fetch("/api/v1/image-models"), "video": _fetch("/api/v1/video-models")}
        out = {k: [m for m in v if is_nsfw(m) and is_paid(m) and is_raster(m)] for k, v in live.items()}
        if out["image"] and out["video"]:
            return out
    except Exception as e:
        print(f"[NanoGPT] live model list failed, using snapshot: {e}")
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_text_models():
    try:
        ids = sorted(m["id"] for m in _fetch("/api/v1/models"))
        if ids:
            return ids
    except Exception as e:
        print(f"[NanoGPT] live text model list failed: {e}")
    return [DEFAULT_TEXT_MODEL]


_MEDIA = _load_media_models()
IMAGE_MODELS = {m["id"]: m for m in _MEDIA["image"]}
VIDEO_MODELS = {m["id"]: m for m in _MEDIA["video"]}
TEXT_MODELS = _load_text_models()


# ---------------------------------------------------------------- conversions

def bytes_to_tensor(b):
    """Encoded image bytes -> ComfyUI IMAGE tensor [1, H, W, C] float 0..1."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(b)))
    if image.mode == "I":
        image = image.point(lambda i: i * (1 / 255))
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def tensor_to_data_urls(images):
    """ComfyUI IMAGE batch -> list of PNG data URLs."""
    urls = []
    for image in images:
        arr = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
        buff = io.BytesIO()
        Image.fromarray(arr).save(buff, "PNG")
        urls.append("data:image/png;base64," + base64.b64encode(buff.getvalue()).decode("utf-8"))
    return urls
