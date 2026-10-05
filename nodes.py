import base64
import io
import json
import re
import time

import requests

from .nanogpt_api import (
    BASE_URL,
    DEFAULT_TEXT_MODEL,
    IMAGE_MODELS,
    TEXT_MODELS,
    VIDEO_MODELS,
    bytes_to_tensor,
    headers,
    raise_for_status,
    tensor_to_data_urls,
)

IMAGE_URL = BASE_URL + "/api/v1/images/generations"
VIDEO_SUBMIT_URL = BASE_URL + "/api/generate-video"
VIDEO_STATUS_URL = BASE_URL + "/api/video/status"
CHAT_URL = BASE_URL + "/api/v1/chat/completions"


def _union(values):
    """Ordered de-duplicated list."""
    out = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


# ------------------------------------------------------------------ image

IMAGE_SIZES = _union(
    ["auto"] + [r for m in IMAGE_MODELS.values() for r in m["supported_parameters"].get("resolutions", [])]
)


class GenerateImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "forceInput": False}),
                "model": (list(IMAGE_MODELS), {"default": next(iter(IMAGE_MODELS))}),
                "size": (IMAGE_SIZES, {"default": "auto"}),
                "seed": ("INT", {"default": 43, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_NAMES = ("image",)
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "NanoGPT"
    FUNCTION = "f"

    def f(self, prompt, model, size, seed, image=None):
        spec = IMAGE_MODELS[model]
        params = spec["supported_parameters"]
        valid_sizes = params.get("resolutions", [])

        json_data = {
            "model": model,
            "prompt": prompt,
            "seed": seed,
            "response_format": "b64_json",
        }
        if size != "auto":
            if size not in valid_sizes:
                raise Exception(f"{model} does not support size '{size}'. Valid: {['auto'] + valid_sizes}")
            json_data["size"] = size

        if image is not None:
            if not spec["capabilities"].get("image_to_image"):
                raise Exception(f"{model} is text-to-image only; disconnect the image input.")
            urls = tensor_to_data_urls(image.contiguous())
            max_in = params.get("max_input_images") or 1
            if len(urls) > max_in:
                raise Exception(f"{model} accepts at most {max_in} input images, got {len(urls)}.")
            json_data["imageDataUrl"] = urls[0]
            if len(urls) > 1:
                json_data["imageDataUrls"] = urls

        r = requests.post(IMAGE_URL, headers=headers(), json=json_data, timeout=300)
        raise_for_status(r)
        item = r.json()["data"][0]
        if item.get("b64_json"):
            raw = base64.b64decode(item["b64_json"])
        else:
            raw = requests.get(item["url"], timeout=120).content
        return (bytes_to_tensor(raw),)


# ------------------------------------------------------------------ video

def _vparams(model):
    return VIDEO_MODELS[model]["supported_parameters"].get("parameters", {})


def _options(key):
    vals = []
    for m in VIDEO_MODELS:
        p = _vparams(m).get(key)
        if p and p.get("options"):
            vals += [str(o["value"]) for o in p["options"]]
    return _union(vals)


AUDIO_KEYS = ("generate_audio", "generateAudio", "enable_audio")
VIDEO_DURATIONS = ["default"] + sorted(_options("duration"), key=lambda d: float(d))
VIDEO_RESOLUTIONS = ["default"] + _options("resolution")
VIDEO_ASPECTS = ["default"] + [a for a in _options("aspect_ratio") if a]
VIDEO_ORIENTATIONS = ["default", "landscape", "portrait"]


def _find(obj, keys):
    """Depth-first search for the first string value under any of `keys`."""
    if isinstance(obj, dict):
        for k in keys:
            if isinstance(obj.get(k), str) and obj[k]:
                return obj[k]
        for v in obj.values():
            hit = _find(v, keys)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find(v, keys)
            if hit:
                return hit
    return None


class GenerateVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (list(VIDEO_MODELS), {"default": next(iter(VIDEO_MODELS))}),
                "duration": (VIDEO_DURATIONS, {"default": "default"}),
                "resolution": (VIDEO_RESOLUTIONS, {"default": "default"}),
                "aspect_ratio": (VIDEO_ASPECTS, {"default": "default"}),
                "orientation": (VIDEO_ORIENTATIONS, {"default": "default"}),
                "audio": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
                "timeout_sec": ("INT", {"default": 900, "min": 60, "max": 3600}),
            },
            "optional": {
                "image": ("IMAGE",),
                "last_image": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "video_url": ("STRING", {"default": ""}),
                "extra_json": ("STRING", {"multiline": True, "default": "{}"}),
            },
        }

    RETURN_NAMES = ("video", "video_url")
    RETURN_TYPES = ("VIDEO", "STRING")
    CATEGORY = "NanoGPT"
    FUNCTION = "f"

    def _build_body(self, prompt, model, duration, resolution, aspect_ratio, orientation, audio, seed,
                    image, last_image, negative_prompt, video_url, extra_json):
        spec = VIDEO_MODELS[model]
        caps = spec["capabilities"]
        params = _vparams(model)
        body = {"model": model, "prompt": prompt}

        def put(key, value):
            p = params.get(key)
            if p is None:
                return
            opts = [str(o["value"]) for o in p.get("options", [])]
            if opts and str(value) not in opts:
                raise Exception(f"{model} does not support {key}='{value}'. Valid: {opts}")
            # Keep the type the model's default uses (some take numbers, some strings).
            body[key] = int(value) if p.get("type") == "number" else str(value)

        if duration != "default":
            put("duration", duration)
        if resolution != "default":
            put("resolution", resolution)
        if aspect_ratio != "default":
            put("aspect_ratio", aspect_ratio)
        if orientation != "default":
            put("orientation", orientation)
        for k in AUDIO_KEYS:
            if k in params:
                body[k] = bool(audio)
        if seed >= 0 and "seed" in params:
            body["seed"] = seed
        if negative_prompt and "negative_prompt" in params:
            body["negative_prompt"] = negative_prompt

        if image is not None:
            if not caps.get("image_to_video"):
                raise Exception(f"{model} does not take a start image.")
            body["imageDataUrl"] = tensor_to_data_urls(image[:1].contiguous())[0]
        elif caps.get("image_to_video") and not caps.get("text_to_video"):
            raise Exception(f"{model} is image-to-video only; connect a start image.")
        if last_image is not None:
            if "last_image" not in params:
                raise Exception(f"{model} does not support last_image.")
            body["last_image"] = tensor_to_data_urls(last_image[:1].contiguous())[0]
        if video_url:
            body["videoUrl"] = video_url
        elif caps.get("video_to_video") and not caps.get("image_to_video"):
            raise Exception(f"{model} needs video_url.")

        extra = json.loads(extra_json or "{}")
        if not isinstance(extra, dict):
            raise Exception("extra_json must be a JSON object.")
        body.update(extra)
        return body

    def f(self, prompt, model, duration, resolution, aspect_ratio, orientation, audio, seed, timeout_sec,
          image=None, last_image=None, negative_prompt="", video_url="", extra_json="{}"):
        from comfy_api.input_impl import VideoFromFile

        body = self._build_body(prompt, model, duration, resolution, aspect_ratio, orientation, audio, seed,
                                image, last_image, negative_prompt, video_url, extra_json)
        r = requests.post(VIDEO_SUBMIT_URL, headers=headers(), json=body, timeout=300)
        raise_for_status(r)
        submit = r.json()

        query = {}
        for key, names in (("requestId", ("requestId", "request_id", "id")), ("runId", ("runId", "run_id")),
                           ("jobId", ("jobId", "job_id")), ("modelSlug", ("modelSlug", "model_slug"))):
            v = _find(submit, names)
            if v:
                query[key] = v
        if not query:
            raise Exception(f"NanoGPT video submit returned no job id: {json.dumps(submit)[:500]}")
        query.setdefault("modelSlug", model)

        url = None
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            s = requests.get(VIDEO_STATUS_URL, headers=headers(), params=query, timeout=120)
            raise_for_status(s)
            data = s.json()
            status = (_find(data, ("status", "state")) or "").lower()
            if any(x in status for x in ("fail", "error", "cancel")):
                raise Exception(f"NanoGPT video job failed: {json.dumps(data)[:1000]}")
            if any(x in status for x in ("complete", "succeed", "success")):
                url = _find(data, ("videoUrl", "video_url", "url", "outputUrl", "output_url"))
                if not url:
                    raise Exception(f"Video completed but no URL found: {json.dumps(data)[:1000]}")
                break
            time.sleep(5)
        if not url:
            raise Exception(f"NanoGPT video timed out after {timeout_sec}s (job {query}).")

        v = requests.get(url, timeout=600)
        raise_for_status(v)
        return (VideoFromFile(io.BytesIO(v.content)), url)


# ------------------------------------------------------------------ prompt (text completion)

DEFAULT_INSTRUCTION = (
    "You are a prompt engineer for AI image and video generation models. "
    "Expand the user's idea into one vivid, detailed prompt: subject, appearance, pose/action, "
    "setting, lighting, camera/lens, composition, style. For video, add motion and camera movement. "
    "Output only the prompt text, no preamble, no quotes."
)


class PromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        default = DEFAULT_TEXT_MODEL if DEFAULT_TEXT_MODEL in TEXT_MODELS else TEXT_MODELS[0]
        return {
            "required": {
                "idea": ("STRING", {"multiline": True}),
                "model": (TEXT_MODELS, {"default": default}),
                "instruction": ("STRING", {"multiline": True, "default": DEFAULT_INSTRUCTION}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 600, "min": 16, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
            },
        }

    RETURN_NAMES = ("prompt",)
    RETURN_TYPES = ("STRING",)
    CATEGORY = "NanoGPT"
    FUNCTION = "f"

    def f(self, idea, model, instruction, temperature, max_tokens, seed):
        json_data = {
            "model": model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": idea},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
        }
        r = requests.post(CHAT_URL, headers=headers(), json=json_data, timeout=300)
        raise_for_status(r)
        text = r.json()["choices"][0]["message"].get("content") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip().strip('"')
        if not text:
            raise Exception(f"{model} returned empty text.")
        return (text,)


NODE_CLASS_MAPPINGS = {
    "NanoGPT - Image Generation": GenerateImage,
    "NanoGPT - Video Generation": GenerateVideo,
    "NanoGPT - Prompt Generator": PromptGenerator,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoGPT - Image Generation": "NanoGPT Image (NSFW, paid)",
    "NanoGPT - Video Generation": "NanoGPT Video (NSFW, paid)",
    "NanoGPT - Prompt Generator": "NanoGPT Prompt Generator",
}
