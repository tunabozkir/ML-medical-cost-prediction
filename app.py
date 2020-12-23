from flask import Flask,request,render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__,template_folder='templates')
loaded_model = pickle.load(open("gb_regressor.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':

         return render_template('index.html')

    if request.method == 'POST':
         input_features = [float(x) for x in request.form.values()]
         feature_values = [np.array(input_features)]
         feature_names = ["age","bmi","children","region","sex_male","smoker_yes"]

         df = pd.DataFrame(feature_values, columns=feature_names)
         for r in ["region"]:
             df[r] = df[r].replace("Northeast",0)
             df[r] = df[r].replace("Northwest",1)
             df[r] = df[r].replace("Southwest",2)
             df[r] = df[r].replace("Northwest",3)


         for g in ["sex_male"]:
             df[g] = df[g].replace("Female",0)
             df[g] = df[g].replace("Male",1)


         for s in ["smoker_yes"]:
             df[s] = df[s].replace("No",0)
             df[s] = df[s].replace("Yes",1)


         prediction = round(loaded_model.predict(df)[0], 2)

         return render_template("index.html",result = prediction)


if __name__ == '__main__':
    app.run()
