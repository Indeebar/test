from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Class labels — should match your training
CLASS_NAMES = ['Contaminated', 'Dry', 'Moisturized']

# Load model
model = load_model("waste_classifier_model")

# Setup FastAPI app
app = FastAPI()

# Allow frontend (optional CORS setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "AgriWaste Quality Classification API is up."}

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))  # match training size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {
        "class": predicted_class,
        "confidence": round(confidence, 4)
    }
