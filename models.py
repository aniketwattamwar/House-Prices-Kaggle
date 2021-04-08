
import numpy as np
import pandas as pd
import yaml
def load_config():
    with open('config.yaml') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
print(config)


def cal_mse(y_true,y_pred):
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    print('mse= '+ str(mse))
    
def LR(train_data,y):
    
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    reg = regressor.fit(train_data, y)
    print(reg.score(train_data,y))
    return reg


def xgb(train_data,y):
    import xgboost as xg
     
    xgb = xg.XGBRegressor(colsample_bytree=config["colsample_bytree"], gamma=config["gamma"], 
                             learning_rate=config["learning_rate"], max_depth=config["max_depth"], 
                              n_estimators=config["n_estimators"],
                             reg_lambda=config["reg_lambda"],
                             subsample=config["subsample"], random_state =config["random_state"],eval_metric='rmse',top_k=config["top_k"])
    xgb.fit(train_data, y)

    return xgb

def generate_csv(pred,model_name):
    
    pred = np.exp(pred)
    pred = pd.DataFrame(pred)
    pred.to_csv(model_name +'_prediction.csv')
    
    
# colsample_bytree=0.5, 
# gamma=1, 
# learning_rate=0.05, max_depth=3, 
# n_estimators=2200,
# reg_lambda=0.8,
# subsample=0.5, random_state =7    

