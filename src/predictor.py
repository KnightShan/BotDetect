import numpy as np

def predict_user(model, input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return "BOT" if prediction[0] == 1 else "NOT A BOT"
