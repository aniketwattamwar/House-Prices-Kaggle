

from utility import *
from models import *
from preprocessing import preprocessing_data
import numpy as np
# from models import *

train_data = load_train_data()
test_data = load_test_data()

output = train_data['SalePrice']
train_data = train_data.drop(['SalePrice'],axis =1)

class Main:
    
    def __init__(self):
        
       
        col_names = ['Alley','PoolQC','Fence','MiscFeature']
        train = preprocessing_data.drop_col(train_data,col_names)
        
        # Training Data Preprocessing
        train = preprocessing_data.encoding(train_data)
        train = preprocessing_data.fill_null_values(train)
        train = preprocessing_data.one_hot_encode(train)
        print(train.shape)
        
        # Testing Data Preprocessing
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

        # Training of models 
        reg = LR(training,y)
        reg_pred = reg.predict(testing)
        generate_csv(reg_pred,'regression')
        print(reg_pred)
        
        
        # XGBoost
        xgb_model = xgb(training,y)
        xgb_pred = xgb_model.predict(testing)
        generate_csv(xgb_pred,'xgb')
        print(xgb_pred)
        

if __name__ == "__main__":
    Main()
    