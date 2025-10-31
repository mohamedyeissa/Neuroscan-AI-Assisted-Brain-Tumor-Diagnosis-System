import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import openai
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import gdown


# Set OpenAI API Key
openai.api_key = ("sk-proj-6bElcnRqmh3ABBMgcEPiaIpxwSPHS8v8PWsB"
                  "_uTPeIPMdX7iCB3PIkmEYP7I6RzY4XsNFSuUaWT3BlbkFJs-bRv0hwH1heht7N34"
                  "fyNusYfdLA1RDgJ9FNnkAHSbNjL9LCZL63QsA8H2wszvcl9pU5VS66wA")

# Load the trained model
@st.cache_resource
def load_model():
    url = 'https://drive.google.com/uc?id=1MvsNwDuDvkceJG7OXsJW7VQGE2se3lFa'
    output = 'model.h5'
    if not os.path.exists(output):  # download only if not present
        gdown.download(url, output, quiet=False)

    try:
        model = tf.keras.models.load_model(output)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Project Title & Description
st.markdown("<h1 style='text-align: center;'>MRI Brain Tumor Classification Using Transfer Learning (Xception)</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center;'>
This app uses transfer learning with the Xception model to classify brain tumors from MRI scans.
It distinguishes between four classes to aid in early diagnosis and medical support.
</div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Patient Info Input
st.subheader("Patient Information")
col1, col2, col3 = st.columns(3)
with col1:
    patient_name = st.text_input("Patient Name", help="Enter the patient's full name")
with col2:
    patient_age = st.number_input("Patient Age", min_value=0, max_value=150, step=1)
with col3:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

col4, col5 = st.columns(2)
with col4:
    family_history = st.radio("Family History of Tumors?", ["Yes", "No"])
with col5:
    prior_treatment = st.text_input("Prior Treatments", help="e.g., surgery, radiation")

# Upload MRI Image
st.subheader("Upload MRI Image")
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

# Symptom Questionnaire
st.subheader("Symptom Questionnaire")
symptoms = {}
questions = [
    ("headache", "Persistent headaches?"),
    ("seizures", "Experienced seizures?"),
    ("vision", "Blurred or altered vision?"),
    ("hearing", "Hearing loss or ringing?"),
    ("balance", "Balance/coordination issues?"),
    ("nausea", "Frequent nausea or vomiting?"),
    ("memory", "Memory loss or confusion?"),
    ("weakness", "Weakness in limbs or face?"),
    ("speech", "Speech/language difficulties?"),
    ("personality", "Personality or behavior changes?")
]

cols = st.columns(2)
for i, (key, question) in enumerate(questions):
    with cols[i % 2]:
        symptoms[key] = st.radio(question, ["Yes", "No"], key=key)

# Submit Button
submit_symptoms = st.button("Submit Symptoms")

# PDF Generation Function
def generate_pdf(text: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40

    lines = text.strip().split('\n')
    for line in lines:
        c.drawString(40, y, line.strip())
        y -= 15
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    buffer.seek(0)
    return buffer

# Run Prediction
if uploaded_file is not None and submit_symptoms:
    if not patient_name or patient_age is None or not patient_gender:
        st.warning("Please complete patient name, age, and gender.")
    else:
        st.subheader("Analysis Results")
        with st.spinner("Analyzing MRI image..."):
            try:
                img = Image.open(uploaded_file).convert("RGB")
                img = img.resize((299, 299))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                predicted_index = np.argmax(prediction)
                predicted_class = class_names[predicted_index]
                confidence = float(np.max(prediction))

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("Input Image")
                    st.image(img, caption="Uploaded MRI", width=250)
                    st.markdown(f"<h4 style='text-align: center; color: #2ecc71;'>Prediction: {predicted_class.upper()}</h4>", unsafe_allow_html=True)

                with col2:
                    st.markdown("Prediction Probabilities")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.barh(class_names, prediction[0], color='teal')
                    ax.set_xlabel("Probability")
                    ax.set_title("Prediction Probabilities")
                    for i, v in enumerate(prediction[0]):
                        ax.text(v + 0.01, i, f"{v:.2f}", va='center')
                    st.pyplot(fig)

                # Report
                st.subheader("Preliminary Report")
                report = f"""
Patient Information:
- Name: {patient_name}
- Age: {patient_age}
- Gender: {patient_gender}
- Family History: {family_history}
- Prior Treatments: {prior_treatment if prior_treatment else 'None'}

Tumor Prediction:
- Predicted Type: {predicted_class.capitalize()}
- Confidence: {confidence:.2f}

Reported Symptoms:
- Headaches: {symptoms['headache']}
- Seizures: {symptoms['seizures']}
- Vision Changes: {symptoms['vision']}
- Hearing Issues: {symptoms['hearing']}
- Balance Problems: {symptoms['balance']}
- Nausea/Vomiting: {symptoms['nausea']}
- Memory Problems: {symptoms['memory']}
- Limb/Facial Weakness: {symptoms['weakness']}
- Speech Issues: {symptoms['speech']}
- Personality Changes: {symptoms['personality']}

AI Remark: Based on the image and symptoms, this may indicate {predicted_class.upper()}.
Further neurological consultation and diagnostics are advised.
                """
                with st.expander("View Detailed Report"):
                    st.markdown(report)

                # OpenAI Recommendation
                gpt_prompt = f"Given patient data (Name={patient_name}, Age={patient_age}, Gender={patient_gender}, Family History={family_history}, Prior Treatments={prior_treatment}, Symptoms={symptoms}, AI Prediction={predicted_class}), provide a medical recommendation."
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a medical assistant providing preliminary recommendations."},
                        {"role": "user", "content": gpt_prompt}
                    ],
                    max_tokens=150
                )
                st.subheader("AI Medical Recommendation")
                st.write(response.choices[0].message.content.strip())

                # PDF Report Download
                pdf_file = generate_pdf(report)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_file,
                    file_name="brain_tumor_report.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
else:
    if not uploaded_file:
        st.info("Please upload an MRI image to proceed.")
    elif not submit_symptoms:
        st.info("Please submit the symptom questionnaire to proceed.")
