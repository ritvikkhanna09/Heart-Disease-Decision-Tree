from sklearn.externals import joblib
import pandas as pd
import numpy as np

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,13)
    loaded_model = joblib.load('finalized_model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]

example_list=[57,0,1,130,236,0,0,174,0,0,1,1,2]

print(ValuePredictor(example_list))