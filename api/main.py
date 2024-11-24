from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
from fastapi.responses import JSONResponse
import os
from fastapi.middleware.cors import CORSMiddleware

os.environ['PYTHONIOENCODING'] = 'utf-8'

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
import sys

# Force UTF-8 encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")



# Load the model and class names
MODEL_PATH = r"C:\Users\Gen\Documents\Text_summerization\Potato-Disease-Classification\models\my_model.keras"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]  # Replace with your actual class names

logging.basicConfig(level=logging.DEBUG)

# Load the trained model
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Helper function to read and preprocess an image
def read_file_as_image(data: bytes) -> np.ndarray:
    logging.debug("Reading file as image")
    image = np.array(Image.open(BytesIO(data)))
    return image

def preprocess_image(image: np.ndarray) -> np.ndarray:
    logging.debug("Preprocessing the image")
    image = tf.image.resize(image, (256, 256))  # Adjust to your model's input size
    image = image / 255.0  # Normalize to [0, 1] range
    return np.expand_dims(image, axis=0)  # Add batch dimension

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logging.debug("Starting prediction process")
        
        # Read and process image
        image_data = await file.read()
        logging.debug(f"File data length: {len(image_data)} bytes")
        image = read_file_as_image(image_data)  # Your image reading function
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image)  # Add your preprocessing here
        logging.debug(f"Image batch shape after preprocessing: {preprocessed_image.shape}")

        # Get predictions
        predictions = MODEL.predict(preprocessed_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        
        # Log the prediction
        logging.debug(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
        
        # Return response
        return JSONResponse(content={
            "class": str(predicted_class),  # Ensure the class name is encoded properly
            "confidence": float(confidence)
        })

    except Exception as e:
        logging.error(f"Error during prediction: {str(e).encode('utf-8', errors='replace')}")
        return JSONResponse(content={"error": str(e).encode("utf-8", errors="replace")}, status_code=500)

    
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

##cd "C:/Users/Gen/Documents/Text_summerization/Potato-Disease-Classification/api"
##$ uvicorn main:app --reload

