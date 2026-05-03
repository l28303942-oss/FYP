import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-pe-segmentation-change-in-production")
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + os.path.join(BASE_DIR, "database", "pe_system.db"),
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "scans")
    RESULT_FOLDER = os.path.join(BASE_DIR, "uploads", "results")
    MODEL_WEIGHTS = os.environ.get(
        "MODEL_WEIGHTS",
        os.path.join(BASE_DIR, "models", "weights", "attention_unet_pe.pth"),
    )
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB (DICOM volumes)
    HOSPITAL_NAME = os.environ.get("HOSPITAL_NAME", "Pulmonary AI Diagnostics Center")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm", "dicom"}
    REMEMBER_COOKIE_DURATION = 60 * 60 * 24 * 14  # 14 days
