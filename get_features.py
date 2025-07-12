import pickle
with open("./model/log_reg_model.pkl", "rb") as filehandler:
    model_info=pickle.load(filehandler)

# Try to get features
try:
    print(model_info["features"])
except KeyError:
    print("Feature names not stored.")

