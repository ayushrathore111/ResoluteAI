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

# Separate features and target
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']


# Standardize the data
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

# # Train SVM
svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
svm_train_acc = accuracy_score(y_train, svm.predict(X_train_scaled))

# # Train Decision Tree
# dt = DecisionTreeClassifier(random_state=42)
# dt.fit(X_train_scaled, y_train)
# y_pred_dt = dt.predict(X_test_scaled)
# dt_train_acc = accuracy_score(y_train, dt.predict(X_train_scaled))

# # Train Gradient Boosting
# gb = GradientBoostingClassifier(random_state=42)
# gb.fit(X_train_scaled, y_train)
# y_pred_gb = gb.predict(X_test_scaled)
# gb_train_acc = accuracy_score(y_train, gb.predict(X_train_scaled))

# # Train k-Nearest Neighbors
# knn = KNeighborsClassifier()
# knn.fit(X_train_scaled, y_train)
# y_pred_knn = knn.predict(X_test_scaled)
# knn_train_acc = accuracy_score(y_train, knn.predict(X_train_scaled))

# Create a DataFrame for test predictions
test_predictions = pd.DataFrame({
    'logreg_prediction': y_pred_logreg,
    'rf_prediction': y_pred_rf,
    'svm_prediction': y_pred_svm,
    # 'dt_prediction': y_pred_dt,
    # 'gb_prediction': y_pred_gb,
    # 'knn_prediction': y_pred_knn
})

# Save predictions to CSV
test_predictions.to_csv('test_predictions.csv', index=False)

# Output train accuracies
print(f"Logistic Regression Train Accuracy: {logreg_train_acc}")
print(f"Random Forest Train Accuracy: {rf_train_acc}")
print(f"SVM Train Accuracy: {svm_train_acc}")
# print(f"Decision Tree Train Accuracy: {dt_train_acc}")
# print(f"Gradient Boosting Train Accuracy: {gb_train_acc}")
# print(f"k-Nearest Neighbors Train Accuracy: {knn_train_acc}")
