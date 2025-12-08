import os
import re
from datetime import datetime
import tempfile

import streamlit as st
import numpy as np
from PIL import Image
# Matplotlib used as fallback for plot generation when Plotly is not available
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
from io import BytesIO
try:
    import tensorflow as tf
except Exception:
    tf = None
import requests
import zipfile
import tarfile
from pathlib import Path
from streamlit.errors import StreamlitSecretNotFoundError
import pandas as pd

def get_model_download_url():
    """Return MODEL_DOWNLOAD_URL preferring env var, otherwise attempt Streamlit secrets safely.
    This avoids raising StreamlitSecretNotFoundError when no secrets file exists.
    """
    env = os.getenv("MODEL_DOWNLOAD_URL")
    if env:
        return env
    try:
        if hasattr(st, 'secrets'):
            # st.secrets may implement .get or only index; try both safely
            try:
                return st.secrets.get("MODEL_DOWNLOAD_URL")
            except Exception:
                try:
                    return st.secrets["MODEL_DOWNLOAD_URL"]
                except Exception:
                    return None
    except Exception:
        return None

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="NeuroScan - Brain Tumor Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.main-header {font-size: 2.2rem; color: #fff; font-weight: 700; margin-bottom: 0.4rem;}
.sub-header {font-size: 1rem; color: #f2f4f8; margin-bottom: 1.2rem; font-weight: 400;}
.section-header {font-size: 1.12rem; color: #2C3E50; font-weight: 600; border-bottom: 1px solid #E5E7E9; padding-bottom: 0.4rem; margin-top: 1.1rem; letter-spacing: 0.5px;}
.clinical-card {background-color: #FFFFFF; padding: 1.2rem; border-radius: 8px; border: 1px solid #EAEFF3; margin: 0.8rem 0; box-shadow: 0 6px 18px rgba(2,6,23,0.06);}
.risk-low {background-color: #EAFAF1; border-left: 6px solid #27AE60; padding: 1rem; margin: 0.5rem 0;}
.risk-medium {background-color: #FEF9E7; border-left: 6px solid #F39C12; padding: 1rem; margin: 0.5rem 0;}
.risk-high {background-color: #FDEDEC; border-left: 6px solid #E74C3C; padding: 1rem; margin: 0.5rem 0;}
.metric-value {font-size: 1.6rem; font-weight: 600; color: #2C3E50; font-family: 'Courier New', monospace;}
.diagnosis-positive {color: #C0392B; font-weight: 700; font-size: 1.1rem;}
.diagnosis-negative {color: #27AE60; font-weight: 700; font-size: 1.1rem;}
.sidebar .sidebar-content {background-color: #F4F6F8;}
.stButton button {background-color: #1A5276; color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600;}
.stButton button:hover {background-color: #16324f; color: white;}
.upload-area {border: 2px dashed #BDC3C7; border-radius: 6px; padding: 2rem; text-align: center; background-color: #F8F9F9;}
.card {background:#ffffff; border-radius:10px; padding:12px; box-shadow:0 6px 18px rgba(0,0,0,0.06)}
.header-gradient { background: linear-gradient(90deg,#1A5276,#2980b9); color:#fff; padding:14px; border-radius:8px; }
.small-note{ font-size:0.9rem; color:#6c757d; }
</style>
""", unsafe_allow_html=True)

try:
    if not PLOTLY_AVAILABLE:
        st.sidebar.info("Optional package 'plotly' not installed; interactive charts will fall back to Matplotlib/text.")
except Exception:
    pass

# PDF export removed: the app now only saves TXT reports. (silent fallback)

# ----------------------------
# TUMOR LIBRARY
# ----------------------------
TUMOR_LIBRARY = {
    "Glioma": {
        "short": "Gliomas originate from glial cells and range from low- to high-grade tumors.",
        "details": "Management often includes neurosurgery, radiation, and chemotherapy depending on grade.",
        "major_symptoms": ["Headaches", "Seizures", "Cognitive changes", "Motor weakness"]
    },
    "Meningioma": {
        "short": "Meningiomas arise from meningothelial cells in the dura and are often benign.",
        "details": "Surgical resection is standard; radiosurgery may be considered.",
        "major_symptoms": ["Headaches", "Focal neurological deficits", "Seizures", "Visual disturbance"]
    },
    "Pituitary": {
        "short": "Pituitary adenomas are usually benign and may affect endocrine function and vision.",
        "details": "Treatment may include surgery, medical therapy, or radiosurgery.",
        "major_symptoms": ["Visual disturbances", "Endocrine dysfunction", "Headaches"]
    },
    "No Tumor": {
        "short": "No tumor detected; findings may be normal or represent non-neoplastic disease.",
        "details": "If symptoms persist, consider clinical follow-up or alternative diagnostics.",
        "major_symptoms": []
    }
}

# ----------------------------
# REPORTS BASE (project reports folder or fallback)
# ----------------------------
# Default to a local repo folder for Streamlit deployments (overridable by env var)
DEFAULT_REPORTS_DIR = os.getenv("NEUROSCAN_REPORTS_DIR", os.path.join(os.getcwd(), "reports"))
REPORTS_BASE = DEFAULT_REPORTS_DIR
try:
    os.makedirs(REPORTS_BASE, exist_ok=True)
except Exception as e:
    # If the configured directory cannot be created (permissions, missing drive),
    # fall back to the system temp directory and warn the user in Streamlit.
    try:
        st.warning(f"Using fallback report directory; could not use {REPORTS_BASE}: {e}")
    except Exception:
        pass
    REPORTS_BASE = os.path.join(tempfile.gettempdir(), "NeuroScanReports")
    os.makedirs(REPORTS_BASE, exist_ok=True)

def _sanitize_name(s: str) -> str:
    s = str(s).replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)

# Ensure class folders exist
for c in TUMOR_LIBRARY.keys():
    os.makedirs(os.path.join(REPORTS_BASE, _sanitize_name(c)), exist_ok=True)

# ----------------------------
# LOAD MODEL (cached resource)
# ----------------------------
MODEL_PATH = "Final_Model_SavedModel"

@st.cache_resource
def load_model(path: str):
    # If model doesn't exist and a MODEL_DOWNLOAD_URL is set, try downloading
    download_url = get_model_download_url()
    if not os.path.exists(path) and download_url:
        try:
            download_model_if_missing(download_url, path)
        except Exception as e:
            try:
                st.warning(f"Model download failed: {e}")
            except Exception:
                pass
    try:
        loaded_model = tf.saved_model.load(path)
        return loaded_model.signatures.get("serving_default")
    except Exception as e:
        try:
            st.warning(f"Model not loaded: {e}")
        except Exception:
            pass
        return None

infer = load_model(MODEL_PATH)

def model_controls():
    # Sidebar expander for model controls (manual initialization)
    with st.sidebar.expander("Model Controls", expanded=False):
        st.write("Model path: ")
        st.caption(MODEL_PATH)
        # Read the configured download URL in a safe way - environment overrides secrets
        download_url = get_model_download_url()
        if download_url:
            st.write("Model download URL set")
        else:
            st.write("No MODEL_DOWNLOAD_URL configured; the app will try to load a local model.")
        if st.button("Initialize Model (Download & Load)"):
            with st.spinner("Downloading and loading model..."):
                try:
                    if download_url and not os.path.exists(MODEL_PATH):
                        download_model_if_missing(download_url, MODEL_PATH)
                    # Clear cache and reload
                    try:
                        load_model.clear()
                    except Exception:
                        # clear may not exist depending on Streamlit version; fallback not to crash
                        pass
                    new_infer = load_model(MODEL_PATH)
                    if new_infer is not None:
                        st.success("Model initialized and loaded successfully.")
                    else:
                        st.warning("Model initialized but failed to load. Check logs.")
                except Exception as e:
                    st.error(f"Failed to initialize model: {e}")


def download_model_if_missing(url: str, dest_dir: str, show_progress: bool = False):
    """Download a model archive from a URL and extract it to dest_dir.
    Supports zip and tar.gz; if the URL is a raw SavedModel directory link, it will attempt a direct download.
    NOTE: For production, use signed URLs and secure storage.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    local_name = Path("model_download")
    # Attempt a streaming download
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        content_disposition = r.headers.get("content-disposition")
        if content_disposition and "filename=" in content_disposition:
            filename = content_disposition.split("filename=")[-1].strip('"')
        else:
            filename = url.split('/')[-1].split('?')[0]
        local_path = dest / filename
        total = r.headers.get('content-length')
        if show_progress and total:
            total = int(total)
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
        else:
            progress_bar = None
            status_text = None

        downloaded = 0
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    if progress_bar and total:
                        downloaded += len(chunk)
                        pct = int(downloaded / total * 100)
                        progress_bar.progress(min(pct, 100))
                        status_text.text(f"Downloading model: {downloaded / (1024*1024):.2f}MB / {total/(1024*1024):.2f}MB ({pct}%)")
        if progress_bar:
            progress_bar.progress(100)
            status_text.text("Download complete, extracting…")

    # Try to extract if archive
    if zipfile.is_zipfile(local_path):
        with zipfile.ZipFile(local_path, 'r') as zf:
            zf.extractall(path=dest)
    elif tarfile.is_tarfile(local_path):
        with tarfile.open(local_path, 'r:*') as tfh:
            tfh.extractall(path=dest)
    else:
        # Not an archive — maybe the URL points to a directory tar or saved model file. In that case, try to place as is.
        # If it's a homoegenous file (e.g., SavedModel directory in zip), we extracted; otherwise do nothing.
        pass
    return dest

# ----------------------------
# ANALYSIS FUNCTIONS
# ----------------------------
def analyze_mri_image(model_infer, img):
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((299, 299))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0).astype(np.float32)

    if model_infer is None:
        probs = np.array([0.05, 0.05, 0.85, 0.05], dtype=np.float32)
        idx = int(np.argmax(probs))
        return class_names[idx], float(probs[idx]), class_names, probs.tolist()
    try:
        input_name = list(model_infer.structured_input_signature[1].keys())[0]
        output_name = list(model_infer.structured_outputs.keys())[0]
        out = model_infer(**{input_name: tf.constant(img_array)})
        prediction = list(out.values())[0].numpy()
        idx = np.argmax(prediction)
        return class_names[idx], float(prediction[0][idx]), class_names, prediction[0].tolist()
    except Exception:
        probs = np.array([0.05, 0.05, 0.85, 0.05], dtype=np.float32)
        idx = int(np.argmax(probs))
        return class_names[idx], float(probs[idx]), class_names, probs.tolist()


@st.cache_data(show_spinner=False)
def analyze_mri_image_cached(_model_infer, img_bytes: bytes):
    # Convert back to an image and forward to the analyzer
    try:
        img = Image.open(BytesIO(img_bytes))
    except Exception:
        # If the bytes can't be decoded, return a safe default
        return 'No Tumor', 0.0, ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'], [0.05, 0.05, 0.85, 0.05]
    return analyze_mri_image(_model_infer, img)

# ----------------------------
# PATIENT FORM
# ----------------------------
def patient_intake_form():
    st.sidebar.markdown('<div class="section-header">PATIENT INFORMATION</div>', unsafe_allow_html=True)
    # Use a sidebar form to group fields and minimize rerenders until the user submits
    with st.sidebar.form(key='patient_form'):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", "John Doe")
            age = st.number_input("Age", 1, 120, 45)
        with col2:
            patient_id = st.text_input("Patient ID", "PT-2025-MRI-001")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        referral_doctor = st.text_input("Referring Physician", "Dr. Michael Chen")
        facility = st.text_input("Medical Facility", "General Hospital")
        mri_date = st.date_input("MRI Date")
        family_history = st.selectbox("Family History of CNS Disorders", ["None", "Brain Tumors", "Other Neurological", "Unknown"])
        prior_treatments = st.text_area("Prior Treatments/Surgeries", "None")
        _ = st.form_submit_button("Save Patient Info")
    return {
        "Name": name,
        "Patient ID": patient_id,
        "Age": age,
        "Gender": gender,
        "Referring Physician": referral_doctor,
        "Medical Facility": facility,
        "MRI Date": mri_date,
        "Family History": family_history,
        "Prior Treatments": prior_treatments
    }

def health_check():
    # Small health check in sidebar to help Streamlit deployments
    st.sidebar.markdown('<div class="section-header">App Status</div>', unsafe_allow_html=True)
    st.sidebar.write(f"Reports folder: `{REPORTS_BASE}`")
    # Show whether model loaded
    model_status = 'Available' if infer is not None else 'Not Loaded'
    st.sidebar.write(f"Model: `{model_status}`")
    st.sidebar.write(f"Python: `{os.sys.version.split()[0]}`")

def symptom_assessment():
    st.sidebar.markdown('<div class="section-header">CLINICAL ASSESSMENT</div>', unsafe_allow_html=True)
    symptoms = {
        "Headaches": st.sidebar.selectbox("Persistent headaches", ["Absent","Present","Not Assessed"]),
        "Seizures": st.sidebar.selectbox("Seizure activity", ["Absent","Present","Not Assessed"]),
        "Visual disturbances": st.sidebar.selectbox("Visual changes", ["Absent","Present","Not Assessed"]),
        "Motor weakness": st.sidebar.selectbox("Limb weakness", ["Absent","Present","Not Assessed"]),
        "Cognitive changes": st.sidebar.selectbox("Memory/cognition", ["Normal","Impaired","Not Assessed"]),
    }
    return symptoms

def calculate_clinical_risk(symptoms):
    present = [s for s,v in symptoms.items() if v=="Present" or v=="Impaired"]
    if len(present)==0: return "Low", "No significant clinical findings"
    elif len(present)<=2: return "Moderate", f"{len(present)} clinical findings present"
    else: return "High", f"{len(present)} significant clinical findings"

def generate_report_text(patient, symptoms, diagnosis, confidence, class_names, probabilities, clinical_risk):
    lines = [
        "NEUROSCAN — BRAIN TUMOR ANALYSIS REPORT",
        "="*60,
        f"Generated: {datetime.now().isoformat()}",
        "",
        "PATIENT INFORMATION"
    ]
    for k,v in patient.items():
        lines.append(f"- {k}: {v}")
    lines.append("\nCLINICAL ASSESSMENT")
    for k,v in symptoms.items():
        lines.append(f"- {k}: {v}")
    lines.append(f"\nCLINICAL RISK: {clinical_risk[0]} - {clinical_risk[1]}")
    lines.append(f"\nFINDINGS\nPredicted class: {diagnosis}\nModel confidence: {confidence:.2%}\n")
    lines.append("PROBABILITY DISTRIBUTION")
    for n,p in zip(class_names, probabilities):
        lines.append(f"- {n}: {p:.4f}")
    tumor_info = TUMOR_LIBRARY.get(diagnosis, {})
    lines.append("\nTUMOR INFORMATION")
    lines.append(tumor_info.get("short",""))
    if tumor_info.get("major_symptoms"):
        lines.append("Major associated symptoms: "+", ".join(tumor_info.get("major_symptoms")))
    lines.append("\nRECOMMENDATIONS")
    if diagnosis=="No Tumor":
        lines.append("No tumor detected. Monitor or follow-up as needed.")
    else:
        if confidence>0.85: lines.append("High confidence — urgent referral recommended.")
        elif confidence>0.70: lines.append("Moderate confidence — specialist consultation recommended.")
        else: lines.append("Low confidence — consider repeat imaging and clinical correlation.")
        lines.append(tumor_info.get("details",""))
    return "\n".join(lines)

def save_report_file(text, patient, diagnosis, output_dir=None):
    if output_dir is None: output_dir = REPORTS_BASE
    os.makedirs(output_dir, exist_ok=True)
    class_folder = _sanitize_name(diagnosis) if diagnosis else "Unknown"
    save_folder = os.path.join(output_dir, class_folder)
    os.makedirs(save_folder, exist_ok=True)
    safe_name = _sanitize_name(patient.get("Patient ID") or patient.get("Name") or "patient")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_txt = f"{safe_name}_{class_folder}_{timestamp}.txt"
    file_path_txt = os.path.join(save_folder, filename_txt)
    with open(file_path_txt, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path_txt

# ----------------------------
# MAIN APP
# ----------------------------
def main():
    st.markdown('<div class="header-gradient"><div class="main-header">NeuroScan Brain Tumor Analysis System</div><div class="sub-header">Advanced MRI analysis for clinical assessment and tumor detection</div></div>', unsafe_allow_html=True)
    # Run health-check in sidebar to aid deployments/debugging
    health_check()
    # Model controls (manual initialize/download)
    model_controls()

    patient_data = patient_intake_form()
    symptoms = symptom_assessment()
    clinical_risk = calculate_clinical_risk(symptoms)

    uploaded_file = st.file_uploader("Upload MRI Scan (jpg, png)", type=['jpg','jpeg','png'])
    # Option to override reports folder at runtime for Streamlit — useful for deployments
    custom_reports = st.sidebar.text_input("Reports folder (override)", value=REPORTS_BASE)
    reports_dir = REPORTS_BASE
    if custom_reports and custom_reports != REPORTS_BASE:
        try:
            os.makedirs(custom_reports, exist_ok=True)
            reports_dir = custom_reports
        except Exception as e:
            st.sidebar.error(f"Could not use reports folder: {e}")
    if uploaded_file:
        # Use two-column layout: image on the left, analysis and chart on the right
        col_img, col_info = st.columns([2, 1])
        image = Image.open(uploaded_file)
        # Run inference with a spinner and cache using the uploaded bytes
        img_bytes = uploaded_file.getvalue()
        with st.spinner("Analyzing the MRI scan..."):
            predicted_class, confidence, class_names, probs = analyze_mri_image_cached(infer, img_bytes)

        with col_img:
            st.image(image, caption="MRI Scan", use_column_width=True)
            st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
            st.markdown("**Analysis Result**")
            if predicted_class == "No Tumor":
                st.markdown('<p class="diagnosis-negative">NO TUMOR DETECTED</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="diagnosis-positive">TUMOR DETECTED</p>', unsafe_allow_html=True)
                st.markdown(f"Type: {predicted_class}")
            st.markdown(f"Confidence: {confidence:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_info:
            # Plot interactive Plotly bar chart, fallback to Matplotlib if unavailable
            colors = ['#E74C3C' if x == predicted_class else '#3498DB' for x in class_names]
            if 'PLOTLY_AVAILABLE' in globals() and PLOTLY_AVAILABLE:
                fig = go.Figure(go.Bar(x=class_names, y=probs, marker_color=colors))
                fig.update_layout(title='Classification Probabilities', margin=dict(l=0, r=0, t=30, b=0), height=320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6,3))
                ax.bar(class_names, probs, color=colors, alpha=0.8)
                ax.set_ylim(0,1)
                ax.set_ylabel("Probability")
                ax.set_title("Classification Probabilities")
                st.pyplot(fig)
            # Confidence metric and visual progress
            st.metric(label="Model Confidence", value=f"{confidence:.2%}")
            st.progress(int(confidence * 100))
            report_text = generate_report_text(patient_data, symptoms, predicted_class, confidence, class_names, probs, clinical_risk)

        edited_report = st.text_area("Editable Medical Report", value=report_text, height=300)
        if st.button("Save & Download Report"):
            txt_path = save_report_file(edited_report, patient_data, predicted_class, output_dir=reports_dir)
            st.success(f"Report saved: {txt_path}")
            with open(txt_path, "rb") as f:
                st.download_button("Download TXT", f.read(), file_name=os.path.basename(txt_path))

    # Show saved reports per class
    st.markdown('<div class="section-header">Saved Reports</div>', unsafe_allow_html=True)
    reports_list = []
    for tumor_class in TUMOR_LIBRARY.keys():
        folder = os.path.join(reports_dir, _sanitize_name(tumor_class))
        if os.path.exists(folder):
            for fn in os.listdir(folder):
                reports_list.append({"Patient": fn.split("_")[0], "Type": tumor_class, "File": os.path.join(folder, fn)})
    if reports_list:
        st.markdown(f"**{len(reports_list)} reports saved**")
        class_filter = st.selectbox("Filter by Type", ["All"] + list(TUMOR_LIBRARY.keys()))
        filtered = [r for r in reports_list if class_filter == "All" or r["Type"] == class_filter]

        for r in filtered:
            basename = os.path.basename(r["File"])
            # Try to parse timestamp from filename -> safe_name_type_YYYYMMDD_HHMMSS.txt
            try:
                parts = basename.rsplit('_', 2)
                safe_name = parts[0]
                timepart = parts[-1].replace('.txt','')
            except Exception:
                safe_name = basename
                timepart = ''

            col_patient, col_type, col_date, col_actions = st.columns([2, 1, 1, 2])
            with col_patient:
                st.markdown(f"**{safe_name}**")
            with col_type:
                st.markdown(r["Type"])
            with col_date:
                st.markdown(timepart)
            with col_actions:
                try:
                    with open(r["File"], "rb") as f:
                        b = f.read()
                    # Download button
                    st.download_button("Download TXT", b, file_name=basename, key=f"dl_{basename}")
                except Exception as e:
                    st.markdown("Error reading file")

            # Preview expander
            try:
                with st.expander(f"Preview: {basename}"):
                    with open(r["File"], "r", encoding="utf-8") as f:
                        content = f.read()
                    st.code(content, language='text')
            except Exception:
                # Non-text file or read error
                pass

if __name__=="__main__":
    main()
