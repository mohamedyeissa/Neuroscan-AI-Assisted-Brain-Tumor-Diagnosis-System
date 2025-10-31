# AI-Assisted-Brain-Tumor-Diagnosis-System

![Project Illustration](Picture1.png)

## Overview
This repository implements a deep learning pipeline for classification of brain tumors from MRI scans. The solution leverages **transfer learning** with the **Xception** architecture to classify MRI images into four classes:

- Glioma  
- Meningioma  
- Pituitary  
- No Tumor  

All experiments, preprocessing, training, evaluation, and result visualization are implemented and documented within the primary Jupyter Notebook (`Brain_Tumor_MRI_Model.ipynb`).

---

## Objectives
- Develop and fine-tune an Xception-based CNN for MRI brain tumor classification.  
- Preprocess and augment MRI images to improve model generalization.  
- Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix.  
- Provide reproducible training and evaluation steps inside a single notebook for research and educational use.  

---

## Dataset
The dataset used for this project is publicly available on **Kaggle**:

**🔗 [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)**  

**Preprocessing steps include:**
- Resize images to **299×299** pixels (Xception input requirement)  
- Normalize pixel values  
- Data augmentation (rotation, shift, flip, zoom)  
- Stratified train/validation/test split to preserve class balance  

(Refer to the notebook for the dataset link and parameters used.)

---

## Model
- **Architecture:** Xception (pretrained on ImageNet; fine-tuned)  
- **Input shape:** 299 × 299 × 3 (RGB)  
- **Loss:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Epochs:** 50  
- **Batch size:** 32  

| Section           | Description                         | Suggested Visual           |
| ----------------- | ----------------------------------- | -------------------------- |
| Patient Intake    | Form for name, age, gender, history | Patient form UI         |
| Symptom Form      | 10-question radio button survey     | Form UI screenshot      |
| Image Upload      | Accepts .jpg/.png MRI scans         | Sample MRI preview      |
| Model Prediction  | Predicts tumor class                | Horizontal bar chart (probability)|
| Report Generation | Detailed report with all info       | Generated PDF example   |
| GPT-Based Advice  | AI-powered medical suggestion       | Text output from OpenAI |
---


## Output Examples
<p align="center"> <img src="images/3_.png" alt="Brain MRI classification illustration" width="600"/> </p>
<p align="center"> <img src="images/4_.png" alt="Brain MRI classification illustration" width="600"/> </p>
<p align="center"> <img src="images/5_.png" alt="Brain MRI classification illustration" width="600"/> </p>



## Performance
Evaluation outputs (available in the notebook):
- Training / validation loss and accuracy curves  
- Confusion matrix and classification report  
- Example predictions with confidence scores  

**Reported validation accuracy:** ≈ **95%**

---

## Tools & Supporting Libraries
- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- pandas  
- scikit-learn  
- matplotlib  
- seaborn  
- Pillow (PIL)  

---

## Repository Structure
├── Brain_Tumor_MRI_Model.ipynb # Notebook: preprocessing, training, evaluation
├── model_xception.h5 # Saved model weights (if included)
├── requirements.txt # Python dependencies
├── results/ # Evaluation plots and metrics
└── README.md
