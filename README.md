# **Hospital Readmission Prediction Project**

## **Project Overview**
This project focuses on predicting hospital readmissions using a dataset containing patient information. The objective is to predict whether a patient will be readmitted within 30 days after discharge based on their demographics, medical history, and previous visits. The analysis involves cleaning the data, training machine learning models, and visualizing the results.

The key tasks include:
- **Data Preprocessing**: Cleaning and transforming the dataset.
- **Exploratory Data Analysis (EDA)**: Visualizing key patterns and relationships.
- **Model Building**: Using classification models such as Logistic Regression and Random Forest to predict readmission.
- **Model Evaluation**: Evaluating model performance using metrics like accuracy, confusion matrix, ROC curve, and AUC.
- **Visualization**: Creating plots to summarize key trends and the model's performance.

## **How to Run the Project**

1. **Clone or Download the Repository**:
   ```bash
   git clone https://github.com/yourusername/hospital-readmission-prediction.git
   ```
   or download the `.ipynb` file.

2. **Install Required Libraries**:
   Install necessary libraries using `pip`:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Load the Dataset**:
   Use pandas to load the dataset:
   ```python
   import pandas as pd
   data = pd.read_csv('hospital_readmission_data.csv')
   ```

4. **Run the Analysis**:
   Open the notebook and execute the following steps to perform data analysis, model building, and evaluation.

5. **Verify Results**:
   After model training, evaluate performance:
   ```python
   accuracy = accuracy_score(y_test, y_pred)
   confusion_matrix = confusion_matrix(y_test, y_pred)
   roc_auc = roc_auc_score(y_test, y_pred)

   print(f"Accuracy: {accuracy}")
   print(f"Confusion Matrix: \n{confusion_matrix}")
   print(f"ROC AUC: {roc_auc}")
   ```

## **Key Steps in Data Processing**

### **1. Data Preprocessing**
   - Handle missing values, encode categorical variables, and scale numerical features.
   - Feature engineering to create new columns for better prediction.

### **2. Exploratory Data Analysis (EDA)**
   - Visualize the distributions of important variables like age, number of previous admissions, and diagnosis codes.

### **3. Model Building**
   - Logistic Regression and Random Forest are used to predict the likelihood of readmission.

### **4. Model Evaluation**
   - Use confusion matrix, accuracy, and ROC AUC to assess model performance.

## **Dependencies**
To run this project, make sure you have:
- **Python 3.x+**
- **Jupyter Notebook** (or another IDE)
- **Required Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## **Sample Queries for Analysis**

Once the model is trained, you can run the following code snippets for evaluation and analysis:

### **Top Predictive Features for Readmission**:
```python
importances = model.feature_importances_
print(importances)
```

### **Hospital Readmission Trends**:
```python
import matplotlib.pyplot as plt
plt.plot(predicted_readmissions, label='Predicted Readmissions')
plt.plot(actual_readmissions, label='Actual Readmissions')
plt.legend()
plt.show()
```

## **Future Enhancements**
- **Advanced Models**: Implement more sophisticated models like Gradient Boosting or XGBoost.
- **Hyperparameter Tuning**: Use techniques like GridSearchCV or RandomSearch to tune the model's parameters.
- **Data Visualization**: Build interactive dashboards with tools like Dash or Plotly.

---

### **Author**
**Omar Zain**  
GitHub: [Omar Zain GitHub](https://github.com/yourusername)

Feel free to contribute or report any issues!
