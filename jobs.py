import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib

# Define paths for the input and output files
data_path = 'merged_dataset.csv'  # Replace with your dataset path

# Load the dataset
df = pd.read_csv(data_path)

# Separate features and label
X = df.drop(columns=['Label'])
y = df['Label']

# Apply MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'minmax_scaler2.joblib')

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X_scaled)

# Save the PCA model
joblib.dump(pca, 'pca_model2.joblib')

# Train Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_pca, y)

# Save the Random Forest model
joblib.dump(clf, 'random_forest_model2.joblib')

print("Models saved successfully.")
