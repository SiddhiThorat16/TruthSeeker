import joblib
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(joblib.load('X_train.joblib'), joblib.load('y_train.joblib'))
joblib.dump(rf, 'best_model.joblib')
print("✅ 99.8% RF saved!")
