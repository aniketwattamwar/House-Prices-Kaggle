
import numpy as np
import pandas as pd

def LR(train_data,y):
    
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    reg = regressor.fit(train_data, y)
    print(reg.score(train_data,y))
    return reg


def xgb(train_data,y):
    import xgboost as xg
    xgb = xg.XGBRegressor(colsample_bytree=0.5, gamma=1, 
                             learning_rate=0.05, max_depth=3, 
                              n_estimators=2200,
                             reg_lambda=0.8,
                             subsample=0.5, random_state =7)
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