import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("summer_winter_dataset.csv")

# Encode categorical target variable
label_encoder = LabelEncoder()
df['Season'] = label_encoder.fit_transform(df['Season'])  # Summer -> 1, Winter -> 0

# Features and target
X = df[['Month', 'Temperature']]
y = df['Season']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
with open("season_model.pkl", "wb") as f:
    pickle.dump((model, label_encoder), f)
