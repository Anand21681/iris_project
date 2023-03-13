from flask import Flask,render_template,request
import pickle
import json
import numpy as np

with open ('artifacts/model.pkl','rb') as file:
    model=pickle.load(file)

with open('artifacts/asset.json','r')as j_file:
    asset=json.load(j_file)

col=asset['columns']



app=Flask(__name__)

@app.route("/")
def fun():
    return render_template("index.html")

@app.route("/get_data" ,methods=['post'])
def data():
    input_data=request.form 
    print(input_data)

    data=np.zeros(len(col))
    data[0]=input_data['sepal_length']
    data[1]=input_data['sepal_width']
    data[2]=input_data['petal_length']
    data[3]=input_data['petal_width']

    result=model.predict([data])

    if result[0]==0:
        iris_value='SETOSA'
    elif result[0]==1:
        iris_value='VERSICOLOR'
    elif result[0]==2:
        iris_value='VERGINICA'

    print(result)
    
    return render_template("index.html",predict_value=iris_value)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000,debug=False)