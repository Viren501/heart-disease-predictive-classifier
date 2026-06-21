# Clinical Heart Disease Predictive Engine

A comprehensive machine learning workspace comparing multiple supervised learning paradigms to classify cardiovascular disease risk. Using an evaluation pipeline, this project preprocesses clinical patient data and systematically benchmarks parametric models, decision trees, random forests, and k-nearest neighbors to maximize diagnostic precision and safety.

## Dataset Description
The analysis utilizes the `Heart_Disease_Prediction.csv` dataset, evaluating key medical attributes:
* **Target Variable:** `Heart Disease` (Encoded binary target: `Presence` or `Absence`).
* **Patient Metrics:** `Age`, `Sex`, `Chest pain type`, `BP` (Blood Pressure), `Cholesterol`, `FBS over 120` (Fasting Blood Sugar).
* **Electrocardiographic & Spatial Metrics:** `EKG results`, `Max HR` (Maximum Heart Rate), `Exercise angina`, `ST depression`, `Slope of ST`, `Number of vessels fluro`, and `Thallium`.

## Methodology
1. **Target Feature Engineering:** Used Scikit-learn's `LabelEncoder` to convert the categorical target variable (`Presence`/`Absence`) into machine-readable format.
2. **Data Partitioning:** Split observations into an 80% training set and a 20% test partition, enforcing fixed random state controls for absolute reproducibility.
3. **Model Prototyping & Architecture Tuning:**
   * **Logistic Regression:** Implemented as a baseline parametric approach.
   * **Decision Tree Classifier:** Configured with a constrained structural growth limit (`max_depth=4`) to limit overfitting.
   * **Random Forest Classifier:** Deployed an ensemble model combining 200 independent trees with balanced shallow depth limits (`max_depth=4`).
   * **K-Nearest Neighbors (KNN):** Implemented an instance-based model parameterized at `n_neighbors=10`.

## Results & Comparative Evaluation
Model classification performance was rigorously calculated on the unseen validation test split, yielding the following core metrics:

* **Logistic Regression:** Surpassed all models in overall classification capability, achieving an accuracy of **`90.74%`**, a precision of `94.44%`, and a recall of `80.95%`.
* **Random Forest Classifier (Ensemble):** Produced highly stable generalization, reaching **`88.88%`** accuracy with a balanced F1-score of `84.21%`.
* **Decision Tree Classifier:** Registered an accuracy of **`87.04%`**. Gini feature importance matrices demonstrated that `Chest pain type` (~41.4%) and `ST depression` (~25.6%) were the strongest analytical drivers for tree path splitting decisions.
* **K-Nearest Neighbors (KNN):** Proved less effective at distinguishing multi-dimensional feature boundaries for this sample size, concluding with a lower accuracy of **`72.22%`**.

## How to Run Locally
1. Clone this repository to your computer workspace environment.
2. Install standard dependent python packages using pip:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
