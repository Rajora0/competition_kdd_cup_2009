
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from joblib import dump
import time

def main(test_size, model_name):
    # Load the data
    data = pd.read_csv('../../data/processed/churn_preprocessed_data.csv')
    data = pd.get_dummies(data, dtype=int)

    # Split the data into features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    # Initialize the AdaBoostClassifier model
    base_model = AdaBoostClassifier(algorithm='SAMME')

    # Define the parameters for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=0)

    # Perform model tuning
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    # Best model found
    best_model = grid_search.best_estimator_
    
    cv_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
    print("Cross-validation ROC AUC score:", cv_score.mean())

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Calculate ROC AUC score on the test set
    test_roc_auc = roc_auc_score(y_test, y_pred)
    print("Test ROC AUC score:", test_roc_auc)

    # Save the model
    dump(best_model, f'../../models/{model_name}.joblib')
    
    # Print the time taken for GridSearchCV
    print("GridSearchCV took {:.2f} seconds".format(end_time - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train model and save')
    parser.add_argument('test_size', type=float, help='Size of the test set (between 0 and 1)')
    parser.add_argument('model_name', type=str, help='Name of the model file to be saved')
    args = parser.parse_args()
    
    main(args.test_size, args.model_name)

