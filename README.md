# Asthma Disease Diagnosis Project

## Overview

This Project uses Python to develop a predictive machine learning model to identify asthma disease risk using patient lifestyle, environmental and allergy factor, medical history, symptoms, clinical measurements. The dataset was sourced from Kaggle and is SYNTHETIC(artificially generated), making it ideal for educational purpose.

The project includes:
- Exploratory Data Analysis(EDA) 
- Data preprocessing
- Handling imbalanced class with SMOTEEN
- Model Training with Logistic Regression, Random Forest, XGBoost and evaluation using precision, Recall, f1-score and ROC-AUC 
- Export the prediction and feature importances for PowerBI visualization.
- Streamlit app for interactive prediction.

## Dataset Source:

Asthma Disease Dataset on Kaggle https://www.kaggle.com/datasets/rabieelkharoua/asthma-disease-dataset/data
Created and shared by (Author): Rabie El Kharoua(2024)

## Files

- asthma.ipynb : Jupyter Notebook with Exploratory data analysis and showing two different results using SMOTE and SMOTEEN using three models.

- asthma.py : load and preprocess data. Clean pipeline code with two models (logistic regression and XGBoost), applying threshold, evaluate model with confusion matrix and get top 15 features from chosen model. Code generate and export csv file to Power BI for visualization.

- asthma_disease_data.csv : Original csv file downloaded from Kaggle
- powerbi_asthma_prediction.csv : Prediction for Power BI
- top_15_feature_importances.csv : Feature importance value to use for visualization.
- merged_csv_asthma_prediction.csv : Enriched dataset by merging two csv files i.e. Prediction and feature with calculating AsthmaRiskScore for visualization. How these factors affects asthma disease.
- ScreenShot - Some screenshot taken from from PowerBI visualization and Streamlit app 
- asthma_prediction_app.py : streamlit app for asthma prediction. It allows user to input patient details and receive prediction. 
- asthma_disease_PowerBi.pbix : This project includes a  Power BI dashboard built using the exported model out. You will need PowerBI desktop to open and explore the .pbix file 

## Dependencies

- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost
- imbalanced-learn
- streamlit
- streamlit run asthma_prediction_app.py

# Conclusion:

Three csv file generated and stored in outputs_powerbi. Logistic Regression is chosen as the final model because for high recall which is better for healthcare screening. In healthcare, recall plays important role than accuracy because it measures how many asthma patients are correctly identified. Precision was low meaning we predicted some false positives.  Note: The dataset is synthetic, meaning results may not fully reflect real world clinical performance.  




