import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

# Preprocessing
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']
X_test = test_data

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data)

# Train Logistic Regression
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)
logreg_train_acc = accuracy_score(y_train, logreg.predict(X_train_scaled))

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
rf_train_acc = accuracy_score(y_train, rf.predict(X_train_scaled))

# Train SVM
svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
svm_train_acc = accuracy_score(y_train, svm.predict(X_train_scaled))

# Create a DataFrame for test predictions
test_predictions = pd.DataFrame({
    'logreg_prediction': y_pred_logreg,
    'rf_prediction': y_pred_rf,
    'svm_prediction': y_pred_svm,
    # 'dt_prediction': y_pred_dt,
    # 'gb_prediction': y_pred_gb,
    # 'knn_prediction': y_pred_knn
})
model_accuracy=pd.DataFrame({
    'Model': ['Logistic', 'RandomForest ','SupportVM'],
    'Accuracy': [logreg_train_acc, rf_train_acc,svm_train_acc]
})

# Save predictions to CSV
test_predictions.to_csv('test_predictions.csv', index=False)
model_accuracy.to_csv('accuracy_models.csv', index=False)