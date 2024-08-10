import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump 




ds =  pd.read_csv('D:\SEKOLAH\KULIAH\SEMESTER 6\PEMROGRAMAN PERANGKAT BERGERAK\TUGAS BESAR\heart.csv')
print(ds.head())


x = ds[['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']]
y = ds['output']

X_train , X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


model_rf = RandomForestClassifier()

model_rf.fit(X_train, y_train)

model_predict_rf = model_rf.predict(X_test)
print("Random Forest", classification_report(y_test,model_predict_rf))

model_predict_app_rf = 'heart_model_predict_rf.pkl'
dump(model_rf, model_predict_app_rf)
model_predict_app = 'heart_model_predict.pkl'
dump(model, model_predict_app)
print(f"Model Berhasil disimpan {model_predict_app}")