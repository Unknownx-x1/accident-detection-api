from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Accident Detection API")

# Load model once when server starts
model = tf.keras.models.load_model("accident_model.keras")

IMG_SIZE = 224

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # Read uploaded image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # Preprocess
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]

        # IMPORTANT: accident = class 0
        if prediction < 0.5:
            result = "accident"
            confidence = 1 - float(prediction)
        else:
            result = "normal"
            confidence = float(prediction)

        return JSONResponse({
            "prediction": result,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )