from flask import Flask, redirect, url_for, request, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# Initialize the Flask app
df=pd.read_csv("diabetes_prediction_dataset.csv")


# Preprocessing using Ordinal Encoder
enc=OrdinalEncoder()
df["smoking_history"]=enc.fit_transform(df[["smoking_history"]])
df["gender"]=enc.fit_transform(df[["gender"]])


# Define Independent and Dependent Variables
x= df.drop("diabetes",axis=1)
y=df["diabetes"]


# 70% data - Train and 30% data - Test
x_train , x_test , y_train, y_test = train_test_split(x,y,test_size=0.3)


# RandomForest Algorithm
model = RandomForestClassifier().fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

app = Flask(__name__)

# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)
# print(type(model))
print('Model loaded. Start serving...')

print('Check http://127.0.0.1:5000/')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('frontend.html')  # Renders an HTML template for the home page

@app.route('/submit', methods=['GET', 'POST'])
def get_data():
    val=None
    
    if request.method == 'POST':
        print("inside post")
        # Get the file from post request
        gender = request.form['gender']
        gender_dict = {'Female':0.0, 'Male':1.0, 'Others':2.0}
        age = request.form.get('age')
        # print(request.form['age'])
        
        hypertension = request.form['hypertension']
        hypertension_dict = {'No':0, 'Yes':1}
        Heart_disease = request.form['Heart_disease']
        Heart_disease_dict = {'No':0, 'Yes':1}
        Smoking = request.form['Smoking']
        Smoking_history_dict = {'never':4.0, 'no Info':0.0, 'current':1.0, 
                            'former':3.0, 'ever':2.0, 'not current':5.0}
        bmi = request.form.get('bmi')
        HbA1c = request.form.get('HbA1c')
        Glucose = request.form.get('Glucose')
        data_list = [
            gender_dict[gender],
            age,
            hypertension_dict[hypertension],
            Smoking_history_dict[Smoking],
            Heart_disease_dict[Heart_disease],
            bmi,
            HbA1c,
            Glucose
        ]
        print(data_list)
        user_data = np.array(data_list)
        print(age)
        # print(type(newpat))
        user_data = np.nan_to_num(user_data, nan=0)
        user_data = user_data.reshape(1, -1)
        result = model.predict(user_data)
        print(result)
        if result == 1:
            val = "Diabetes"
        else:
            val = "No Diabetes"
            
            
    return render_template('frontend.html',value=val)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
