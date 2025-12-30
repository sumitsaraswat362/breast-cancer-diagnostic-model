# breast-cancer-diagnostic-model
Logistic regression–based classification of breast tumor samples using clinically relevant features.


## Project Overview

This project explores a machine learning approach for classifying breast tumor samples as malignant or benign using structured clinical features. The focus is on understanding how simple, interpretable statistical models can perform in a medical classification setting using routinely measured tumor characteristics.

The project was developed as a self-initiated learning and research exercise to build foundational experience at the intersection of data science and healthcare.

---

## Dataset

The model is trained and evaluated using the Breast Cancer Wisconsin (Diagnostic) dataset, accessed via scikit-learn.

- Number of samples: 569  
- Number of features: 30  
- Target classes:  
  - 0 → Malignant  
  - 1 → Benign  

Each sample represents measurements computed from digitized images of fine needle aspirate (FNA) of breast masses.

Examples of features include:
- Mean tumor radius  
- Texture variation  
- Concavity and compactness  
- Perimeter and area statistics  

This dataset is widely used for benchmarking binary classification models in medical machine learning.

---

## Methodology

A Logistic Regression model was selected due to its interpretability and suitability for binary classification tasks, particularly in clinical contexts where transparency is important.

Workflow:
1. Load and inspect the dataset  
2. Perform basic exploratory analysis and visualization  
3. Split data into training and testing sets  
4. Train a logistic regression classifier  
5. Evaluate performance using accuracy, precision, recall, and F1-score  
6. Analyze feature importance via model coefficients  
7. Test predictions using a sample patient input  

---

## Results

The trained model achieved:
- Accuracy of approximately 96–97% on the test set  
- Strong precision and recall for both malignant and benign classes  

Features with higher coefficient magnitudes include:
- Texture-related measurements  
- Mean radius  
- Worst concavity  
- Worst compactness  

These features are commonly discussed in clinical literature related to tumor morphology and diagnostic assessment.


### ROC Curve (Cross-Validated)

The ROC curve below visualizes the model’s ability to distinguish between malignant and benign tumors across different classification thresholds.  
It was generated using stratified 5-fold cross-validation to ensure robustness across class distributions.

![ROC Curve](results/roc_curve.png)

- Mean ROC-AUC: **0.9917

---

## Example Prediction

The model supports predictions on new patient data by returning:
- A classification label (Malignant or Benign)  
- A probability score representing model confidence  

A synthetic example patient input is included to demonstrate the end-to-end prediction pipeline.

---

## Limitations

- The dataset is a benchmark dataset and does not represent real hospital deployment data  
- No external validation dataset was used  
- The model is intended for educational and exploratory purposes only  
- This project must not be used for real medical diagnosis or clinical decision-making  

---

## Future Improvements

- Train and evaluate the model using real-world open medical datasets (CSV format)  
- Add feature scaling and cross-validation  
- Compare performance with other models (SVM, Random Forest, etc.)  
- Build a simple interface or API for non-technical users  
- Explore explainability techniques for medical machine learning  

---

## Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- Jupyter Notebook  

---

## Author Notes

This repository reflects my current understanding of applied machine learning and will evolve as I gain deeper exposure to medical datasets, evaluation methods, and model interpretation techniques.
