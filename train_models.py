from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import pandas as pd
import numpy as np

print("=== Phase 3: 3-Model Comparison ===")

# Load Phase 2 data
print("Loading preprocessed data...")
X_train = joblib.load('X_train.joblib')
X_test = joblib.load('X_test.joblib')
y_train = joblib.load('y_train.joblib')
y_test = joblib.load('y_test.joblib')

models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='linear', random_state=42, probability=True)
}

results = {}
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {'accuracy': acc, 'f1': f1}
    print(f"Accuracy: {acc:.3f} | F1: {f1:.3f}")
    
    if acc > best_score:
        best_score = acc
        best_model = model

# Results Table
df_results = pd.DataFrame(results).T
print("\n=== MODEL COMPARISON ===")
print(df_results.round(4))

print(f"\n🏆 BEST: {max(results, key=results.get)} (Acc: {best_score:.3f})")

# Save BEST model
joblib.dump(best_model, 'best_model.joblib')
print("\n✅ Best model saved: best_model.joblib")

# Detailed report for best model
print("\n=== BEST MODEL REPORT ===")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=['Real', 'Fake']))
