import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="NeuroScan - Brain Tumor Analysis",
    page_icon="🔬",
    layout="wide"
)

# ----------------------------
# SIMPLIFIED MODEL LOADING
# ----------------------------
@st.cache_resource
def load_model():
    try:
        # Try loading the .keras file
        model = tf.keras.models.load_model(r"D:\DEPI\app\ai_brain_tumor\Final_Model.keras")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_model()

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def analyze_mri_image(img):
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction))
    
    return predicted_class, confidence, class_names, prediction[0]

# ----------------------------
# MAIN APPLICATION
# ----------------------------
def main():
    st.title("NeuroScan Brain Tumor Analysis")
    st.write("Upload an MRI scan for analysis")
    
    if model is None:
        st.error("❌ Model not loaded. Please check the model file.")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Choose an MRI image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                predicted_class, confidence, class_names, probabilities = analyze_mri_image(image)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Diagnosis Result")
                    if predicted_class == "No Tumor":
                        st.success(f"✅ {predicted_class}")
                    else:
                        st.error(f"⚠️ {predicted_class} Detected")
                
                with col2:
                    st.subheader("Confidence")
                    st.metric("Confidence Level", f"{confidence:.1%}")
                    st.progress(confidence)
                
                # Probability chart
                st.subheader("Probability Distribution")
                fig, ax = plt.subplots()
                bars = ax.bar(class_names, probabilities, color=['red' if x == predicted_class else 'blue' for x in class_names])
                ax.set_ylabel('Probability')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Recommendations
                st.subheader("Clinical Recommendations")
                if predicted_class == "No Tumor":
                    st.info("""
                    - Routine follow-up recommended
                    - Continue monitoring if symptoms persist
                    - Consult neurologist for any concerns
                    """)
                else:
                    st.warning(f"""
                    **Urgent neurosurgical consultation recommended for {predicted_class}**
                    - Additional imaging studies needed
                    - Neurological examination required
                    - Consider biopsy for definitive diagnosis
                    - Discuss treatment options with specialist
                    """)

if __name__ == "__main__":
    main()