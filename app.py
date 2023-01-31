from flask import Flask ,request,render_template
import pickle
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('rf_model.pkl','rb'))

def reccomend_system(humidity):
    metadata = pd.read_csv('recommend.csv', low_memory=False)
    newdata=metadata.drop(['Machine','Production','Defect'],axis=1)
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(newdata, newdata)
    indices = pd.Series(metadata.index, index=metadata['Humidity']).drop_duplicates()
    idx=indices[humidity]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores)
    sim_scores = sim_scores[-10:]
    para_indices = [i[0] for i in sim_scores]
    return newdata['Temperature'].iloc[para_indices[5]],newdata['Humidity'].iloc[para_indices[5]],newdata['Current'].iloc[para_indices[5]],newdata['Flow'].iloc[para_indices[5]],newdata['Job Temp'].iloc[para_indices[5]],newdata['Voltage'].iloc[para_indices[5]]


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        current = float(request.form['current'])
        humidity = float(request.form['humidity'])
        temperature = float(request.form['temperature'])
        flow = float(request.form['flow'])
        jobTemp = float(request.form['jobTemp'])
        voltage=float(request.form['voltage'])
       


        data=np.array([[current,humidity,temperature,flow,jobTemp,voltage]])
        prediction=model.predict(data)

        if(prediction==0):
            prediction_txt="Good To go !! No Defect Will Be Produced"
        elif(prediction==1):
            prediction_txt="Sorry Porosity Defect will Happen"
        elif(prediction==2):
            prediction_txt="Sorry Tungsten Inclusion Defect might happen"
        print(prediction)
        return render_template ("predict.html",prediction_text="{}".format(prediction_txt),predval=prediction)

    else:
        return render_template("predict.html")


@app.route('/recommend',methods=['POST','GET'])
def recommend():
    if request.method == 'POST':
        humidity = int(request.form['humidity'])
        temp,hum,curr,flow,jobTemp,vol,=reccomend_system(humidity)
        # print(a,b,c,d,e,f)
        return render_template("recommend.html",temp=temp,hum=hum,curr=curr,flow=flow,jobTemp=jobTemp,vol=vol)
    else:
        return render_template("recommend.html")    
@app.route('/dashboard',methods=['POST','GET'])
def dashboard():
    return render_template("dashboard.html") 

if __name__=="__main__":
    app.run(debug=True)                   