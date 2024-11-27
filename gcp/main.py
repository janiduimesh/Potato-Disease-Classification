from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
from flask import jsonify  

model = None
interpreter = None
input_index = None
output_index = None

class_names = ["Early Blight", "Late Blight", "Healthy"]

BUCKET_NAME = "disease-classification-bucket"  # GCP bucket name

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/potatoes.h5",
            "/tmp/potatoes.h5",
        )
        model = tf.keras.models.load_model("/tmp/potatoes.h5")

    # Get image from request
    image = request.files["file"]
    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256))  # Image resizing
    )
    image = image / 255  # Normalize the image

    img_array = tf.expand_dims(image, 0)  # Add batch dimension
    predictions = model.predict(img_array)

    print("Predictions:", predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    # Ensure the response is JSON serializable
    return ({"class": predicted_class, "confidence": float(confidence)})

## after every change in here, need deploy again "gcloud functions deploy predict --runtime python39 --trigger-http --allow-unauthenticated --region us-central1 --project disease-classification-442812"

