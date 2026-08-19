"""Image processing utilities for preparing images for OpenAI vision API."""

import base64
import io
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}


def is_supported_image(filename: str) -> bool:
    """Check if a filename has a supported image extension."""
    return Path(filename).suffix.lower() in SUPPORTED_FORMATS


def get_supported_files(filenames: list[str]) -> list[str]:
    """Filter filenames to only supported image formats."""
    return [f for f in filenames if is_supported_image(f)]


def load_and_encode_image(image_bytes: bytes, filename: str) -> str | None:
    """Load image from bytes, ensure it's valid, and return base64-encoded string.

    Does not unnecessarily resize to preserve small table text.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verify it's a valid image
        # Re-open after verify (verify closes the stream)
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary (e.g. RGBA PNGs, palette images)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")

        # Encode to the format the user uploaded
        suffix = Path(filename).suffix.lower()
        fmt = "JPEG"
        media_type = "image/jpeg"
        if suffix == ".png":
            fmt = "PNG"
            media_type = "image/png"
        elif suffix == ".webp":
            fmt = "WEBP"
            media_type = "image/webp"

        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return b64

    except Exception as e:
        logger.error("Failed to process image %s: %s", filename, e)
        return None


def get_media_type(filename: str) -> str:
    """Return the media type string for the given filename."""
    suffix = Path(filename).suffix.lower()
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    return mapping.get(suffix, "image/jpeg")


def build_image_content(b64_data: str, filename: str) -> dict:
    """Build the image content dict for the OpenAI API message."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{get_media_type(filename)};base64,{b64_data}",
            "detail": "high",
        },
    }
