import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.metrics import precision_recall_curve
from imblearn.combine import SMOTEENN, SMOTETomek
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score, roc_curve

#General setting before starting project
csv_path = "asthma_disease_data.csv"
output_dir = "outputs_powerbi"
threshold = 0.3

os.makedirs(output_dir, exist_ok=True)

# Load and clean data

def load_data():
    df = pd.read_csv(csv_path)
    
    #check for missing values
    missing_value =df.isnull().sum()
    missing_value = missing_value[missing_value> 0]. sort_values(ascending=False)
    print("\n Missing values:", missing_value)
    
    #Drop irrelevant columns
    df = df.drop(columns =[col for col in ["PatientID", "DoctorInCharge"] if col in df.columns])
    X = df.drop("Diagnosis", axis =1)
    y = df["Diagnosis"]
    return X, y
#X, y =load_data() to see data is loaded properly

#function for Preprocess data: Split, SMOTE(balance data) and Scale
def preprocess_data(X, y):
    #Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)
    #Balance data
    smoteen = SMOTEENN(random_state=42)
    X_train_balance, y_train_balance = smoteen.fit_resample(X_train, y_train)
    #standardize feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balance)
    X_test_scaled = scaler.transform(X_test)
    return X_train_balance, X_test, y_train_balance, y_test, X_train_scaled,X_test_scaled

#Function to train models(Logistic regression and XGBoost)

def train_models(X_train_scaled, X_train_balance, y_train_balance):
    #logistic regression with GridSearch
    log_reg = LogisticRegression(max_iter = 1000, random_state =42)
    log_params = {'C':[0.01,0.1,1, 10],
                  'penalty':['l2'],
                  'solver':['lbfgs']
                  }
    grid_log =GridSearchCV(log_reg, log_params, cv=5, scoring='roc_auc', n_jobs =-1)
    grid_log.fit(X_train_scaled, y_train_balance)
    best_log = grid_log.best_estimator_
    print("Best Logistic Regression ROC-AUC(CV):", grid_log.best_score_)
    print("Best Logistic Regression Params:", grid_log.best_params_)

    #Random Forest with gridsearch

    random_forest = RandomForestClassifier(random_state=42)
    random_forest_params = {
      'n_estimators':[100, 200],
       'max_depth':[None, 10,20],
       'min_samples_split':[2, 5]
       }
      
    grid_random_forest= GridSearchCV(random_forest, random_forest_params, cv = 5, scoring = 'roc_auc', n_jobs=-1)
    grid_random_forest.fit(X_train_balance, y_train_balance)
    best_random_forest = grid_random_forest.best_estimator_
    print("Best Random Forest ROC-AUV(CV):", grid_random_forest.best_score_, "\nParameters:", grid_random_forest.best_params_)

    return best_log, best_random_forest

#Predict Probabilities and apply threshold

def apply_threshold(model, X, threshold):
    y_scores_prob =  model.predict_proba(X)[:,1]
    y_pred = (y_scores_prob>= threshold).astype(int)
    return y_pred, y_scores_prob

#Evaluation of models

def evaluate_model(name, y_true, y_pred):
    print(f"\n {name} Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\n {name} Classification Report:")
    print(classification_report(y_true, y_pred, digits =3))

#Top 15 features from Logistic Regression

def get_top_15_features(best_log, feature_names):
    coefficients = best_log.coef_[0]

    feature_importance = pd.Series(np.abs(coefficients), index = feature_names)
    top_15_features= feature_importance.sort_values(ascending=False).head(15)

    print("Top 15 features from Logistic regression:\n",(top_15_features))
    return top_15_features

#function to save csv file in PowerBI
def save_powerbi(X_test, y_test, log_preds, log_probs, top_15_features):
    #save prediction in powerbi
    df_preds = pd.DataFrame({"TrueLabel": y_test.values,
                             "LogReg_pred": log_preds,
                             "LogReg_prob": log_probs
                             })
    df_preds.to_csv(os.path.join(output_dir,"powerbi_asthma_prediction.csv"), index=False)
    print("Asthma Prediction saved to powerbi_asthma_prediction.csv")
    
    #select top_15 features save in powerbi
    df_features = top_15_features.reset_index()
    df_features.columns =["Feature", "Importance"]
    df_features.to_csv(os.path.join(output_dir,"top_15_feature_importances.csv"),index=False)
    print("Top 15 feature Importances saved to top_15_feature_importances.csv")
    

# function to run code
def run_code():
    print("Asthma Diesase Code started running..")
    X, y = load_data()
    print("Data Shape:", X.shape)
    print("Class distribution:", Counter(y))

    X_train_balance, X_test, y_train_balance, y_test, X_train_scaled,X_test_scaled = preprocess_data(X,y)
    print("Data Split, SMOTEEN applied on training set and scaled.")
    
    #for streamlit app (save the model and scaler)
    scaler= StandardScaler()
    scaler.fit(X_train_balance)
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved for Stramlit app.")

    best_log, best_random_forest = train_models(X_train_scaled, X_train_balance, y_train_balance)
    print("Models Trained for evaluation.")

    joblib.dump(best_log, "logistic_model.pkl")
    print("Model saved for Streamlit app.")

    log_preds, log_probs = apply_threshold(best_log, X_test_scaled, threshold)
    rf_preds, rf_probs = apply_threshold(best_random_forest, X_test, threshold)

    evaluate_model("Logistic Regression", y_test, log_preds)
    evaluate_model("Random Forest", y_test, rf_preds)

    top_15_features= get_top_15_features(best_log, X.columns)
    save_powerbi(X_test, y_test, log_preds, log_probs, top_15_features)

    # Load predictions and feature importances
    df_pred = pd.read_csv(os.path.join(output_dir, 'powerbi_asthma_prediction.csv'))
    df_imp = pd.read_csv(os.path.join(output_dir, 'top_15_feature_importances.csv'))

    # Load original test features
    X, y = load_data()
    _, X_test, _, y_test, _, _ = preprocess_data(X, y)

   # Reset index to align with df_pred
    X_test = X_test.reset_index(drop=True)
    df_pred = df_pred.reset_index(drop=True)

    # Create importance dictionary
    importance_dict = dict(zip(df_imp['Feature'], df_imp['Importance']))

    #Extract top feature names
    top_features = top_15_features.index.tolist()
    
    # Merge top features into prediction DataFrame
    df_pred = pd.concat([df_pred, X_test[top_features].reset_index(drop=True)], axis=1)

    # Calculate WeightedScore
    df_pred['AsthmaRiskScore'] = sum(
            df_pred[feature] * importance_dict.get(feature, 0)
            for feature in importance_dict
            )

    # Save enriched dataset
    df_pred.to_csv(os.path.join(output_dir, 'merged_csv_asthma_prediction.csv'), index=False)
    print("Enriched predictions saved to merged_csv_asthma_prediction.csv")

    print("\nCode running successful.")

# Run the script
if __name__ == "__main__":
 run_code()


