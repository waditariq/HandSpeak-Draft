import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data_dict = pickle.load(open('data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels']).astype(int)  # Ensure labels are integers

# Verify dataset labels
unique, counts = np.unique(labels, return_counts=True)
print(f"✅ Training Labels Distribution: {dict(zip(unique, counts))}")

# Debug: Print first 10 labels to verify correct mapping
print(f"🔍 First 10 Labels: {labels[:10]} (Should be 0=A, 1=B, 2=L)")

# Ensure stratify works by checking class distribution
if min(counts) < 2:
    print("⚠ Warning: At least one class has less than 2 samples. Stratified splitting may fail.")
    stratify_param = None
else:
    stratify_param = labels

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=stratify_param, random_state=42
)

# Train Random Forest Model with class balancing
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(x_train, y_train)

# Evaluate Training Accuracy
train_predictions = model.predict(x_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"✅ Training Set Accuracy: {train_accuracy * 100:.2f}%")

# Evaluate Test Accuracy
y_predict = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_predict)
print(f"✅ Test Set Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
with open("model.p", "wb") as f:
    pickle.dump({'model': model}, f)

print("✅ Model saved as model.p")
