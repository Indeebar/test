from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
import os
import zipfile

# ✅ Unzip model once at startup (Render doesn't like big folders in repo)
if not os.path.exists("waste_classifier_model"):
    with zipfile.ZipFile("waste_classifier_model.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# ✅ Load model in SavedModel format (exported from Colab using model.export)
model = load_model("waste_classifier_model")

# ✅ Class labels (must match training order)
CLASS_NAMES = ['Contaminated', 'Dry', 'Moisturized']

# ✅ FastAPI app setup
app = FastAPI()

# ✅ Enable CORS for web frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Health check route
@app.get("/")
def read_root():
    return {"message": "AgriWaste Quality Classification API is up and running!"}

# ✅ Prediction route
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))  # match training input
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {
        "class": predicted_class,
        "confidence": round(confidence, 4)
    }
