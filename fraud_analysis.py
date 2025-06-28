# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Display the first 5 rows
print("First 5 rows of the data:")
print(df.head())

# Get a summary of the dataset (columns, data types, non-null values)
print("\nDataset Information:")
df.info()

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

from sklearn.preprocessing import StandardScaler

# Create a scaler
scaler = StandardScaler()

# Scale 'Amount' and 'Time'
df['scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop original columns
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Reorder: scaled_Time, scaled_Amount, V1–V28..., Class
columns = ['scaled_Time', 'scaled_Amount'] + [col for col in df.columns if col not in ['scaled_Time', 'scaled_Amount', 'Class']] + ['Class']
df = df[columns]

# Preview after scaling
print("\nScaled Data Preview:\n", df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Split into features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split (stratify to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train baseline logistic regression
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)

# Predict on test set
y_pred_baseline = baseline_model.predict(X_test)

# Evaluate baseline model
print("\n--- Baseline Model Evaluation (Without SMOTE) ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_baseline))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_baseline))

from imblearn.over_sampling import SMOTE

# Check original training class distribution
print("\nOriginal class distribution in training set:\n", y_train.value_counts())

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Show new class distribution after SMOTE
print("\nResampled class distribution:\n", y_train_resampled.value_counts())

# Train new logistic regression model on balanced data
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_train_resampled, y_train_resampled)

# Predict on original test set
y_pred_final = final_model.predict(X_test)

# Evaluate the SMOTE-enhanced model
print("\n--- Final Model Evaluation (After SMOTE) ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

# Create a copy of test set with predictions
results_df = X_test.copy()
results_df['actual_class'] = y_test
results_df['predicted_class'] = y_pred_final

# Save the results as CSV
results_df.to_csv('fraud_detection_results.csv', index=False)
print("\n✅ Results exported to fraud_detection_results.csv")