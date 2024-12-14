from django.apps import AppConfig
from api.model.model_utils import OCRModel


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"

    def ready(self):
        # Loading the model once when the app runs to avoid multiple loadings during runtime
        print("Initializing OCR model...")
        self.ocr_model = OCRModel()