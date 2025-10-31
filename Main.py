import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import openai
from io import BytesIO

# ----------------------------
# CONFIGURATION
# ----------------------------
openai.api_key = "YOUR_API_KEY_HERE"  # replace or set as env var
MODEL_URL = "https://drive.google.com/uc?id=1MvsNwDuDvkceJG7OXsJW7VQGE2se3lFa"
MODEL_PATH = "model.h5"

# ----------------------------
# MODEL LOADING
# ----------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model weights...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# ----------------------------
# IMAGE PREDICTION
# ----------------------------
def predict_image(model, image_path):
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    img = Image.open(image_path).convert("RGB")
    img = img.resize((299, 299))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction))

    print(f"\n--- Prediction Results ---")
    print(f"Predicted Class: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.2f}")

    # Optional: show probabilities
    print("\nClass Probabilities:")
    for c, p in zip(class_names, prediction[0]):
        print(f"  {c}: {p:.3f}")

    return predicted_class, confidence, class_names, prediction[0]

# ----------------------------
# AI RECOMMENDATION
# ----------------------------
def generate_recommendation(patient_data, symptoms, predicted_class):
    gpt_prompt = (
        f"Given patient data: {patient_data}, "
        f"and symptoms: {symptoms}, "
        f"AI prediction: {predicted_class}. "
        "Provide a short, responsible medical recommendation."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical assistant providing preliminary, non-diagnostic recommendations."},
                {"role": "user", "content": gpt_prompt}
            ],
            max_tokens=150
        )
        recommendation = response.choices[0].message.content.strip()
        return recommendation
    except Exception as e:
        print(f"⚠️ Error generating AI recommendation: {e}")
        return None

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    print("=== MRI Brain Tumor Classification ===")

    # Load model
    model = load_model()

    # Input section (can be replaced with a script or UI later)
    image_path = input("\nEnter MRI image path (jpg/png): ").strip()
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Collect patient info
    patient_data = {
        "Name": input("Patient Name: ").strip(),
        "Age": input("Patient Age: ").strip(),
        "Gender": input("Gender (Male/Female/Other): ").strip(),
        "Family History": input("Family History of Tumors? (Yes/No): ").strip(),
        "Prior Treatments": input("Prior Treatments (if any): ").strip()
    }

    # Collect symptom info
    symptoms = {
        "Headache": input("Persistent headaches? (Yes/No): ").strip(),
        "Seizures": input("Experienced seizures? (Yes/No): ").strip(),
        "Vision Issues": input("Blurred or altered vision? (Yes/No): ").strip(),
        "Hearing Issues": input("Hearing loss or ringing? (Yes/No): ").strip(),
        "Balance Problems": input("Balance/coordination issues? (Yes/No): ").strip(),
        "Nausea": input("Frequent nausea or vomiting? (Yes/No): ").strip(),
        "Memory Issues": input("Memory loss or confusion? (Yes/No): ").strip(),
        "Weakness": input("Weakness in limbs or face? (Yes/No): ").strip(),
        "Speech Issues": input("Speech/language difficulties? (Yes/No): ").strip(),
        "Personality Changes": input("Personality or behavior changes? (Yes/No): ").strip()
    }

    # Prediction
    predicted_class, confidence, class_names, probabilities = predict_image(model, image_path)

    # AI recommendation
    recommendation = generate_recommendation(patient_data, symptoms, predicted_class)
    if recommendation:
        print("\n--- AI Medical Recommendation ---")
        print(recommendation)

    # Optional: visualize prediction probabilities
    plt.barh(class_names, probabilities, color="teal")
    plt.xlabel("Probability")
    plt.title("Prediction Probabilities")
    plt.show()

    print("\nProcess completed successfully.")

