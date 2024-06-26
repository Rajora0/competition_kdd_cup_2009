import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from joblib import dump

def etc_importances(features, outcome):
    model = ExtraTreesClassifier(n_estimators=200, criterion='entropy', verbose=0)
    model.fit(features, np.array(outcome).ravel())
    
    importances = model.feature_importances_
    variables = np.array(features.columns)
    indices = np.argsort(importances)[::-1]
    importances = importances[indices]
    variables = variables[indices]
    
    return variables, importances

def function_sum_importances(variables, importances):
    sum_importances = pd.DataFrame(columns=['Variable', 'Sum_Importance'])
    for i in range(importances.size):
        sum_importance = importances[:(i+1)].sum()
        this_variable = pd.DataFrame([[variables[i], sum_importance]], columns=['Variable', 'Sum_Importance'])
        sum_importances = pd.concat([sum_importances, this_variable], ignore_index=True)
    return sum_importances

def keep_vars(features, sum_importances, threshold):
    keep_vars = list(sum_importances[sum_importances.iloc[:, 1] <= threshold].iloc[:, 0])
    features = features.loc[:, keep_vars]
    return features

def main(input_data, output_model):
    # Carregar dados
    features = pd.read_csv(input_data)
    outcome_upselling = pd.read_csv('../../data/raw/orange_small_train_upselling.labels', header=None).rename(columns={0: 'Upselling'})
    
    # Preprocessamento
    features_u = pd.get_dummies(features)
    variables, importances = etc_importances(features_u, outcome_upselling)
    sum_importances = function_sum_importances(variables, importances)
    features_u = keep_vars(features_u, sum_importances, threshold=0.99)
    preprocessed_data = pd.concat([features_u, outcome_upselling], axis=1)
    preprocessed_data = pd.get_dummies(preprocessed_data, dtype=int)
    
    # Dividir os dados
    X = preprocessed_data.drop('Upselling', axis=1)
    y = preprocessed_data['Upselling']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo
    base_model = GradientBoostingClassifier()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Avaliar modelo
    cv_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    y_pred = best_model.predict(X_test)
    test_roc_auc = roc_auc_score(y_test, y_pred)
    
    # Salvar modelo
    dump(best_model, output_model)
    
    print("Cross-validation ROC AUC score:", cv_score)
    print("Test ROC AUC score:", test_roc_auc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upselling Prediction")
    parser.add_argument("--input_data", type=str, help="Path to input data")
    parser.add_argument("--output_model", type=str, help="Path to save trained model")
    args = parser.parse_args()
    
    main(args.input_data, args.output_model)
