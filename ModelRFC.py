from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
import numpy as np
import pickle

data_dict = pickle.load(open('./data.pickle', 'rb'))
max_length = max(len(item) for item in data_dict['data'])
data = np.array([item + [0] * (max_length - len(item)) for item in data_dict['data']])
#data = np.asarray(data_dict['data'])

labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Random Forest Classifier
model_rfc = RandomForestClassifier()
model_rfc.fit(x_train, y_train)
y_predict_rfc = model_rfc.predict(x_test)
accuracy_rfc = accuracy_score(y_test, y_predict_rfc)
print(f"Random Forest Accuracy: {accuracy_rfc*100}")

# Support Vector Machine Classifier
model_svc = SVC()
model_svc.fit(x_train, y_train)
y_predict_svc = model_svc.predict(x_test)
accuracy_svc = accuracy_score(y_test, y_predict_svc)
print(f"SVM Accuracy: {accuracy_svc*100}")

# HistGradient Boosting Classifier
model_hgbc = HistGradientBoostingClassifier()
model_hgbc.fit(x_train, y_train)
y_predict_hgbc = model_hgbc.predict(x_test)
accuracy_hgbc = accuracy_score(y_test, y_predict_hgbc)
print(f"HistGradient Boosting Accuracy: {accuracy_hgbc*100}")

# Determine the best model
best_model = None
best_accuracy = 0

if accuracy_rfc > best_accuracy:
    best_model = model_rfc
    best_accuracy = accuracy_rfc

if accuracy_svc > best_accuracy:
    best_model = model_svc
    best_accuracy = accuracy_svc

if accuracy_hgbc > best_accuracy:
    best_model = model_hgbc
    best_accuracy = accuracy_hgbc

# Save the best model using pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)