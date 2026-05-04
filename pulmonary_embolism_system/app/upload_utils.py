"""Filename / extension helpers for scan uploads (handles .nii.gz, etc.)."""
from __future__ import annotations

# Kept in sync with app.config Config.ALLOWED_EXTENSIONS
RASTER_PDF_PREVIEW = frozenset(
    {
        "png",
        "jpg",
        "jpeg",
        "jfif",
        "pjp",
        "pjpeg",
        "webp",
        "bmp",
        "tif",
        "tiff",
        "gif",
        "ico",
    }
)


def normalized_extension(filename: str) -> str:
    """
    Return lowercase extension, including 'nii.gz' for NIfTI.
    """
    if not filename or not filename.strip():
        return ""
    name = filename.strip()
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return "nii.gz"
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[1].lower()
