

from utility import *
from models import *
from preprocessing import preprocessing_data
import numpy as np
# from models import *

train_data = load_train_data()
test_data = load_test_data()

output = train_data['SalePrice']
train_data = train_data.drop(['SalePrice'],axis =1)

# def load_config():
#     with open('config.yaml') as file:
#         config = yaml.safe_load(file)
#     return config
# 
config = load_config()


class Main:
    
    def __init__(self):
        
       
        # col_names = ['Alley','PoolQC','Fence','MiscFeature','BsmtFinSF2','BsmtFinSF1','TotalBsmtSF','GrLivArea']
        col_names = config["col_names"]
        train = preprocessing_data.drop_col(train_data,col_names)
        print(train.shape)
        
        # Training Data Preprocessing
        train = preprocessing_data.encoding(train_data)
        train = preprocessing_data.fill_null_values(train)
        train = preprocessing_data.one_hot_encode(train)
        print(train.shape)
        
        # Testing/Validation Data Preprocessing
        test = preprocessing_data.encoding(test_data)
        test = preprocessing_data.fill_null_values(test)
        test = preprocessing_data.one_hot_encode(test)
        print(test.shape)
        
        
        training, testing = train.align(test, join='inner', axis=1)
        print(training.shape)
        print(testing.shape)
        
        
        # output log transform
        y = np.log(output)
        print(y)
        
        #splitting train, test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(training, y, test_size=0.2, random_state=42)
        print(X_test.shape)
        
        # Training of models 
        reg = LR(X_train,y_train)
        reg_pred = reg.predict(X_test)
        
        # Calculate MSE for Regression Model
        cal_mse(reg_pred,y_test)
        
        
        xgb_model = xgb(X_train,y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        # Calculatae MSE for xgb
        cal_mse(xgb_pred,y_test)
        
        
        if config["model_name"]=='Regression':
            pred = reg.predict(testing)
            generate_csv(pred,'outputs/regression')
        # print(reg_pred)
        
        if config["model_name"]== 'XGBoost':
            x_pred = xgb_model.predict(testing)
            generate_csv(x_pred,'outputs/xgb')
        # print(xgb_pred)
        
        
        # Cross validation
        # PCA- Dimensionality Reduction
        # grid search
        
if __name__ == "__main__":
     
    Main()
    