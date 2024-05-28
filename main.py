import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Task 1: Apply K-means clustering and display predicted clusters
st.title('Machine Learning and Data Analysis App')

st.header('Task 1: K-means Clustering')

train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')
st.subheader('Original Training Dataset')
st.write(train_data)

X_train = train_data.drop('target', axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X_train_scaled)

train_data['cluster'] = kmeans.labels_


X_test_scaled = scaler.transform(test_data)
test_data['cluster'] = kmeans.predict(X_test_scaled)


st.subheader('Dataset with Predicted Clusters')
st.write(test_data)
# test_data.to_csv('test_with_clusters.csv', index=False)

# Task 2: Train and test classification models
st.header('Task 2: Train and Test Classification Models')
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

# Train Decision Tree
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
model_accuracy=pd.DataFrame({
    'Model': ['Logistic', 'RandomForest ','SupportVM'],
    'Accuracy': [logreg_train_acc, rf_train_acc,svm_train_acc]
})

# Save predictions to CSV
# test_predictions.to_csv('test_predictions.csv', index=False)
st.subheader('Models Accuracy')
st.write(model_accuracy)
st.subheader('Model Predictions')
st.write(test_predictions)





# Task 3: Datewise analysis
st.header('Task 3: Datewise Analysis')

raw_data = pd.read_excel('rawdata.xlsx')
# raw_data['datetime'] = pd.to_datetime(raw_data['date'] + ' ' + raw_data['time'])
raw_data['date'] = raw_data['date']
raw_data['location'] = raw_data['location'].str.lower()

inside_activities = raw_data[raw_data['location'] == 'inside']
outside_activities = raw_data[raw_data['location'] == 'outside']

inside_duration = inside_activities.groupby('date').size().reset_index(name='total_duration_inside')
outside_duration = outside_activities.groupby('date').size().reset_index(name='total_duration_outside')

duration_summary = pd.merge(inside_duration, outside_duration, on='date', how='outer').fillna(0)

activity_count = raw_data.groupby(['date', 'activity']).size().unstack(fill_value=0)
activity_count = activity_count.reset_index()
activity_count.columns = ['date', 'num_picking', 'num_placing']

st.subheader('Datewise Total Duration')
st.write(duration_summary)

st.subheader('Datewise Activity Count')
st.write(activity_count)
