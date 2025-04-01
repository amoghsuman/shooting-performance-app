import pandas as pd
import joblib
import gzip
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("shooting_data.csv")
df.drop(columns=["Shooter_ID"], inplace=True)

# Encode categoricals
categorical_cols = ["Experience_Level", "Handedness", "Lighting_Conditions", "Training_Type"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define columns
target_col = "Shot_Accuracy (%)"
feature_cols = [col for col in df.columns if col != target_col]
numerical_cols = [col for col in feature_cols if col not in categorical_cols]

# Scale only features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split
X = df[feature_cols]
y = df[target_col]  # Don't scale y!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Eval
y_pred = model.predict(X_test)
print(f"📊 MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"📊 MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"📊 R²:  {r2_score(y_test, y_pred):.4f}")

# Save artifacts
def save_gz(obj, filename):
    with gzip.open(filename, 'wb') as f:
        joblib.dump(obj, f)

save_gz(model, "shooting_performance_model.pkl.gz")
save_gz(scaler, "scaler.pkl.gz")
save_gz(label_encoders, "label_encoders.pkl.gz")
save_gz(feature_cols, "feature_names.pkl.gz")
