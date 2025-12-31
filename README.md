# breast-cancer-diagnostic-model
Machine learning–based classification of breast tumor samples using clinically relevant diagnostic features.

---

## Project Overview

This project investigates the use of machine learning models for classifying breast tumor samples as **malignant** or **benign** based on structured clinical features extracted from medical imaging data.  

The primary goal is not only to achieve high predictive performance, but to evaluate models through a **clinical lens**, where interpretability, robustness, and minimizing life-threatening errors are critical.

This work was developed as a self-initiated research project to gain hands-on experience at the intersection of **machine learning, statistics, and healthcare diagnostics**.

---

## Dataset

The models are trained and evaluated using the **Breast Cancer Wisconsin (Diagnostic)** dataset, accessed via `scikit-learn`.

- **Total samples:** 569  
- **Number of features:** 30  
- **Target classes:**  
  - `0` → Malignant  
  - `1` → Benign  

Each sample consists of measurements computed from digitized images of **fine needle aspirate (FNA)** of breast masses.

Representative features include:
- Mean tumor radius  
- Texture and smoothness  
- Concavity and concave points  
- Perimeter and area statistics  

This dataset is a canonical benchmark in diagnostic machine learning research and is primarily used for controlled algorithmic comparison rather than deployment-level validation.

---

## Methodology

The analysis follows a structured and reproducible workflow:

1. Dataset loading and inspection  
2. Exploratory data analysis  
3. Train–test split with stratification  
4. Model training and hyperparameter selection  
5. Performance evaluation using multiple metrics  
6. Cross-validation to ensure robustness  
7. Visualization of diagnostic performance  

Initial experimentation focused on **Logistic Regression** due to its interpretability, followed by a comparative study with more complex models.

All preprocessing, model fitting, and evaluation steps were performed strictly within the training folds to prevent information leakage between training and evaluation data.

---

## ROC Curve Analysis (Cross-Validated)

The Receiver Operating Characteristic (ROC) curve illustrates the model’s ability to distinguish between malignant and benign tumors across classification thresholds.

To ensure reliability, the ROC-AUC score was computed using **5-fold stratified cross-validation**, preserving class balance across folds.

![ROC Curve](results/roc_curve.png)

- **Mean ROC-AUC:** 0.9917  

This indicates strong separability between the two classes.

---

## Comparative Model Analysis

Multiple machine learning models were evaluated using **identical data splits and evaluation metrics**:

- Logistic Regression  
- Support Vector Machine (RBF Kernel)  
- Random Forest  
- Multi-Layer Perceptron (Neural Network)

![Model Comparison](results/model_comparison.png)

### SHAP Feature Importance (Random Forest)

SHAP (SHapley Additive exPlanations) was used to interpret the Random Forest
model by quantifying the contribution of each feature to malignant
predictions. The summary plot highlights clinically relevant features such
as concavity, radius, and texture as dominant drivers of malignancy
classification.

![SHAP Summary](results/shap_summary_random_forest_malignant.png)

### Key Findings

While Random Forest achieved the highest overall accuracy, the SVM model achieved the highest recall for malignant cases on the internal test split, minimizing false negatives under the selected decision threshold.

In clinical diagnostics:
- A **False Negative** (malignant predicted as benign) can delay or prevent life-saving treatment  
- A **False Positive** may lead to additional tests, which is comparatively less dangerous  

For this reason, **recall for malignant cases was prioritized over raw accuracy**.

**Conclusion:**  
From a clinical risk perspective, the **Support Vector Machine** emerges as the most reliable diagnostic model in this study.

---

## Example Prediction

The trained models support predictions on unseen patient data by returning:
- A predicted class label (Malignant / Benign)  
- A probability score reflecting model confidence  

A synthetic patient example is included in the notebook to demonstrate the full prediction pipeline.

---

## External Validation and Dataset Shift Analysis

To assess the generalizability of the trained models beyond the development cohort, an external validation attempt was conducted using independently sourced open breast cancer datasets.

Multiple candidate datasets were evaluated, including publicly available CSV datasets from Kaggle and the UCI Machine Learning Repository. During this process, two critical challenges were identified:
	1.	Dataset Identity and Leakage Risk
Certain external files (e.g., commonly shared data.csv versions on Kaggle) were found to be numerically identical to the original Wisconsin Diagnostic Breast Cancer (WDBC) dataset after column alignment. Using such datasets for external validation would introduce severe data leakage, as the model would be evaluated on samples it had effectively already seen during training. These datasets were therefore explicitly excluded from validation.
	2.	Dataset Shift and Feature Incompatibility
Truly independent datasets (e.g., Coimbra Breast Cancer Dataset, SEER-based clinical datasets) exhibited substantial differences in feature definitions, data distributions, and patient cohorts. These datasets contain anthropometric, biochemical, or demographic variables rather than imaging-derived morphological features, making them incompatible with the trained models without extensive feature re-engineering.

When tested on mismatched datasets, model performance degraded or became uninterpretable, which is an expected outcome under covariate shift and domain mismatch. This result highlights a key limitation in medical machine learning: models trained on highly curated benchmark datasets may not generalize across heterogeneous clinical populations without domain adaptation or retraining.

Key takeaway:
The absence of a suitable, feature-aligned external cohort prevents valid external performance claims. Rather than forcing unreliable validation, this limitation is documented transparently to avoid misleading conclusions.

This analysis reinforces the importance of dataset provenance, feature consistency, and cohort alignment when developing machine learning models for clinical decision support.


## Limitations

- The dataset is a benchmark dataset and does not reflect real hospital deployment data  
- No feature-aligned external cohort was available that would allow valid out-of-distribution evaluation without introducing data leakage or     domain mismatch   
- Feature extraction is predefined and not learned from raw images  
- This project is intended for **educational and research purposes only**  
- The models must **not** be used for real medical diagnosis  

---

## Future Work

Planned improvements include:
- Extended explainability analysis, including class-conditional SHAP stability, interaction effects, and clinician-oriented feature         attribution summaries    
- Feature-to-clinical interpretation mapping  
- Evaluation on additional open medical datasets (CSV format)  
- Ensemble modeling and threshold optimization  
- Improved documentation aligned with scientific manuscripts  

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

This repository reflects an evolving understanding of applied machine learning in medical contexts.  
Future updates will focus on **interpretability, clinical relevance, and research-grade evaluation practices**.
