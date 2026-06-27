 RareDetect - AI Based Brain Tumor Detection

Overview
RareDetect is a deep learning-based application developed to detect brain tumors from MRI images. 
The project compares the performance of three models: **VGG16**, **ResNet50**, and **RareDetect**,
a custom CNN model developed for this project.
The models were trained and evaluated using MRI datasets, and their performance was analyzed using validation accuracy, training curves, and confusion matrices. The application is deployed using Streamlit, allowing users to upload an MRI image, receive a prediction, view the confidence score, and download a prediction report.



Features

- MRI Image Upload
- Brain Tumor / No Tumor Prediction
- Comparison of VGG16, ResNet50 and RareDetect
- Confidence Score Display
- Confusion Matrix Visualization
- Downloadable Prediction Report
- Simple Streamlit-Based User Interface

---

Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Matplotlib
- Pillow

---

Project Structure

```text
RareDetect/
│── app.py
│── final_year_project1.ipynb
│── vgg16_model.h5
│── resnet50_model.h5
│── raredetect_model.h5
│── dataset.zip
│── Brain_Tumor_Report.pdf
│── cm_raredetect.png
│── README.md
```

---

Workflow

1. Upload MRI Image
2. Image Preprocessing
3. Feature Extraction
4. Prediction using Deep Learning Models
5. Display Tumor / No Tumor Result
6. Show Confidence Score
7. Download Prediction Report

---

## Models Used

- VGG16 (Transfer Learning)
- ResNet50 (Transfer Learning)
- RareDetect (Custom CNN)

---

 How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Future Scope

- Improve model performance using larger MRI datasets.
- Extend the system for multi-class tumor classification.
- Deploy the application on cloud platforms.
- Integrate Explainable AI techniques for better model interpretation.



## Developed By

Samruddhi H. Sabale
Rushikesh Shiraskar 
Rutuja Londe 
piyush Borade

Department of Computer Engineering

JSPM's Jaywantrao Sawant College of Engineering, Hadapsar, Pune
