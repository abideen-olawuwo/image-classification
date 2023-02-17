import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_C=1.0.bin'




with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('converted')


@app.route('/predict', methods=['POST'])
def predict():
 prospect_id = request.get_json()

 X = dv.transform([prospect_id])
 y_pred = model.predict_proba(X)[0, 1]
 converted = y_pred >= 0.5
 result = {
  'converted_probability' : float(y_pred),
  'converted' : bool (converted)
 }

 return jsonify(result)



if __name__ == "__main__":
 app.run(debug=True, host='0.0.0.0', port=3000)


