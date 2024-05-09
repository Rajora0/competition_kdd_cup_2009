import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from pycaret.classification import *

def preprocess_data(num_threshold, cat_threshold):
    # Read feature and outcome data
    features = pd.read_csv('../../data/raw/orange_small_train.data', sep='\t', na_filter=True)
    outcome = pd.read_csv('../../data/raw/orange_small_train_churn.labels', header=None).rename(columns={0: 'Churn'})

    # Divide variables into numeric and categorical
    all_vars = np.array(features.columns)
    num_vars = np.array(all_vars[:190])
    cat_vars = np.array(all_vars[190:])

    # Replace empty cells with NaN
    features = features.replace('', np.nan)

    # Convert numeric columns to float and categorical columns to category
    for col in num_vars:
        features[col] = features[col].astype('float')
    for col in cat_vars:
        features[col] = features[col].astype('category')

    # Check for empty entries and calculate threshold
    empty_entries_per_column = features.isna().sum(axis=0)
    num_entries = len(features)
    threshold = num_threshold/100

    # Keep only columns with less or equal to the threshold of empty entries
    keep_vars = np.array(features.columns[(empty_entries_per_column <= (num_entries * threshold))])
    num_vars = [elem for elem in num_vars if elem in keep_vars]
    cat_vars = [elem for elem in cat_vars if elem in keep_vars]

    # Fill missing values
    for col in num_vars:
        col_mean = features[col].mean()
        features[col] = features[col].fillna(col_mean)
    for col in cat_vars:
        features[col] = features[col].cat.add_categories('missing')
        features[col] = features[col].fillna('missing')

    # Filter categorical variables based on threshold
    n_categories_per_feature = features[cat_vars].apply(lambda x: len(set(x)))
    cat_vars = np.array(n_categories_per_feature[n_categories_per_feature < cat_threshold].index)

    # Select final features
    features = features[list(num_vars) + list(cat_vars)]

    return features, outcome

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <numeric_threshold> <categorical_threshold> <output_filename>")
        sys.exit(1)

    num_threshold = int(sys.argv[1])
    cat_threshold = int(sys.argv[2])
    output_filename = sys.argv[3]

    preprocessed_features, outcome = preprocess_data(num_threshold, cat_threshold)

    # Save preprocessed data to CSV
    preprocessed_data = pd.concat([preprocessed_features, outcome], axis=1)
    preprocessed_data.to_csv('../../data/processed/'+ output_filename, index=False)

    print(f"Preprocessed data saved to {output_filename}")
