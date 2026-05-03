import os
import sys

from flask import Flask

# Allow `from ml.xxx` imports from project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.config import Config
from app.extensions import db, login_manager
from app.services.seed import seed_if_empty


def create_app(config_class=Config):
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )
    app.config.from_object(config_class)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)
    os.makedirs(os.path.join(app.config["BASE_DIR"], "uploads", "reports"), exist_ok=True)
    os.makedirs(os.path.join(app.config["BASE_DIR"], "database"), exist_ok=True)
    os.makedirs(os.path.join(app.config["BASE_DIR"], "models", "weights"), exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    from app.routes.auth import auth_bp
    from app.routes.main_routes import main_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()
        seed_if_empty()

    @app.context_processor
    def inject_globals():
        from flask_login import current_user

        theme = "light"
        if current_user.is_authenticated:
            theme = getattr(current_user, "theme", None) or "light"
        return {
            "hospital_name": app.config.get("HOSPITAL_NAME", "Hospital"),
            "user_theme": theme,
        }

    return app
