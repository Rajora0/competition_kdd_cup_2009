train_model_churn.py:
- `test_size`: Size of the test set as a float between 0 and 1. This determines the proportion of the dataset used for testing.
- `model_name`: Name of the model file to be saved.
Example usage:

```
python train_model_churn.py 0.2 best_model_churn
```

This command will split the dataset into 80% training and 20% testing, train the model, and save it as best_model_churn.joblib.

train_model_upselling.py:

- `--input_data`: Path to the preprocessed data file (e.g., "../../data/processed/churn_preprocessed_data.csv").
- `--output_model`: Path to save the trained model file (e.g., "../../models/best_model_upselling.joblib").

Example usage:

```
python train_model_upselling.py --input_data ../../data/processed/churn_preprocessed_data.csv --output_model ../../models/best_model_upselling.joblib
```

train_model_appentency.py:

- `--input_data`: Path to the preprocessed data file (e.g., "../../data/processed/churn_preprocessed_data.csv").
- `--output_model`: Path to save the trained model file (e.g., "../../models/best_model_appetency.joblib").

Example usage:

```
python train_model_appentency.py --input_data ../../data/processed/churn_preprocessed_data --output_model ../../models/best_model_appentency.joblib
```
