# SVM-FR-glass-identification-for-waste-sorting
Comparing SVM and Random Forest models to accurately classify glass types for recycling automation.

# Glass Classification for Recycling using Machine Learning

## Project Overview
A comparative study implementing and optimizing Support Vector Machine (SVM) and Random Forest classifiers to accurately identify glass types based on their chemical composition. This project aims to improve automation in glass recycling facilities by reducing misclassification and increasing purity rates.

**Keywords:** Machine Learning, Recycling, Sustainability, SVM, Random Forest, R, Hyperparameter Tuning, Classification

## Business Understanding
Glass is 100% recyclable, but inefficient sorting leads to contamination and landfilling. This project tackles the core problem of accurately automating glass sorting to improve recycling rates and reduce environmental impact.

## Data
The project uses the [UCI Glass Identification Dataset](https://archive.ics.uci.edu/dataset/42/glass+identification). It contains 214 instances with 9 features measuring the chemical composition (e.g., Na, Mg, Al) and refractive index of glass samples. The target variable is the glass type (7 classes).

## Methods
1.  **Data Preprocessing:** Handled class imbalance, applied and compared Z-Score, Min-Max, and SoftMax normalization.
2.  **Exploratory Data Analysis (EDA):** Conducted correlation analysis and visualized feature distributions.
3.  **Modeling:** Built and optimized two models:
    - **Support Vector Machine (SVM):** Optimized using grid search for parameters `C` and `gamma`.
    - **Random Forest (RF):** Optimized using cross-validation for parameters `mtry` and `ntree`.
4.  **Evaluation:** Models were evaluated based on accuracy, precision, recall, F1-score, and AUC.

## Results
- The **tuned SVM model** achieved **97.7% accuracy**, a significant improvement from the baseline (83.7%).
- The **tuned Random Forest model** achieved **76.9% accuracy** and provided valuable feature importance insights, identifying Al, Mg, and RI as the most discriminative features.
- A detailed comparison of the trade-offs between the two models (accuracy vs. interpretability, speed vs. robustness) is provided.
