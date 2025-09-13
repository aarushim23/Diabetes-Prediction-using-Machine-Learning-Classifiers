# Diabetes Prediction using Machine Learning Classifiers

## Abstract

The application of machine learning has emerged as a pioneering approach in the prognosis and diagnosis of diseases within healthcare. By leveraging data-driven algorithms, clinicians and researchers can enhance early detection, improve treatment recommendations, and optimize patient outcomes. In this study, we investigate the performance of four distinct classifiers—**K-Nearest Neighbour (KNN)**, **Support Vector Machine (SVM)**, **Decision Tree (DT)**, and **Random Forest (RF)**—for diabetes prediction using a clinical dataset of 2000 female patients. After preprocessing steps involving outlier removal, missing data imputation, and feature selection, the models were trained and evaluated. The classifiers achieved accuracies of **81.16% (KNN)**, **84.84% (SVM)**, **77.16% (DT)**, and **89.83% (RF)**, with Random Forest outperforming the others in both accuracy and AUC (0.95).

This work has been published in the 2025 19th International Conference on Ubiquitous Information Management and Communication (IMCOM) 

---

## Dataset Overview

The dataset comprises diagnostic records of **2000 female patients**, with **684 diabetic** and **1316 non-diabetic** cases. Each record includes eight diagnostic features and one outcome variable:

| Feature                    | Description                                |
| -------------------------- | ------------------------------------------ |
| Pregnancies                | Number of pregnancies                      |
| Glucose                    | Plasma glucose concentration after 2 hours |
| Blood Pressure             | Diastolic blood pressure (mm Hg)           |
| Skin Thickness             | Triceps skin fold thickness (mm)           |
| Insulin                    | 2-hour serum insulin (µU/ml)               |
| BMI                        | Body Mass Index (kg/m²)                    |
| Diabetes Pedigree Function | Family history of diabetes                 |
| Age                        | Patient’s age                              |

The target variable is binary: **0 = Non-diabetic, 1 = Diabetic**.

---

## Methodology

### Data Preprocessing

* **Outlier Removal** – To preserve statistical integrity and avoid bias.
* **Handling Missing Data** – Missing values in glucose, blood pressure, skin thickness, insulin, and BMI were imputed using mean/median values.
* **Feature Selection** – Correlation-based techniques to retain relevant predictors.

### Exploratory Data Analysis (EDA)

* Visualizations: overlapping histograms, correlation matrices, and heatmaps.
* Insights: glucose, BMI, and age showed strongest correlation with diabetes.

### Statistical Analysis

* Regression, ANOVA, eigenvalue decomposition, and Wilks’ lambda were applied to identify feature significance.
* Results confirmed glucose, BMI, and age as key predictors.

### Machine Learning Models

Four models were implemented using scikit-learn:

* **KNN** – distance-based classifier.
* **SVM** – maximizing class separation in high-dimensional space.
* **Decision Tree** – hierarchical partitioning of features.
* **Random Forest** – ensemble of decision trees with bagging.

Evaluation metrics included **accuracy, precision, recall, specificity, F1-score, and ROC-AUC**.

---

## Results

* **Random Forest** achieved the **highest accuracy (89.83%)** and **AUC (0.95)**.
* **Decision Tree** exhibited the weakest performance with 77.16% accuracy.
* Comparative results demonstrate the robustness of ensemble methods over standalone classifiers.

| Model         | Accuracy | AUC Score |
| ------------- | -------- | --------- |
| KNN           | 81.16%   | 0.92      |
| SVM           | 84.84%   | 0.92      |
| Decision Tree | 77.16%   | 0.83      |
| Random Forest | 89.83%   | 0.95      |

---

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

---

## Usage

Run the Jupyter notebooks to reproduce the analysis:

```bash
jupyter lab
```

Notebooks included:

* `01_data_preprocessing.ipynb`
* `02_exploratory_data_analysis.ipynb`
* `03_statistical_analysis.ipynb`
* `04_model_training_and_evaluation.ipynb`

---

## Conclusion

This study highlights the significant role of machine learning in healthcare analytics. Ensemble methods, particularly Random Forest, offer robust and accurate diagnostic support in diabetes prediction tasks. Future work will explore advanced deep learning models and feature engineering strategies to further improve diagnostic performance.

---

## Citation

If you use this repository in your research, please cite:

```
R. Chauhan, A. Mishra, R. J. Mani, E. Yafi and M. F. Zuhairi, "An Analytical Paradigm for Exploration of Diabetes Using Machine Learning," 2025 19th International Conference on Ubiquitous Information Management and Communication (IMCOM), Bangkok, Thailand, 2025, pp. 1-8, doi: 10.1109/IMCOM64595.2025.10857504. keywords: {Support vector machines;Machine learning algorithms;Machine learning;Medical services;Diabetes;Drug discovery;Decision trees;Medical diagnostic imaging;Random forests;Diseases;Artificial Intelligence;ML;Healthcare Databases;Diabetes;Classifiers},
```
