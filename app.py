import flask
from flask import Flask,render_template,request
import numpy
import pickle
import sklearn
from sklearn.linear_model import LinearRegression
with open("Simple_Linear_Regression_model.pkl","rb") as f:
    model=pickle.load(f)

app=Flask(__name__)

@app.route("/")
def fun():
    return render_template("index.html")
@app.route("/predict",methods=['GET','POST'])
def prediction():
    val=[float(i) for i in request.form.values()]
    val=numpy.array(val)
    sol=model.predict([val])[0]
    return render_template("index.html",prediction_text=sol)
if __name__=="__main__":
    app.run(debug=True)