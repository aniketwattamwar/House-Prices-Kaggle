

from utility import *
from preprocessing import preprocessing_data
# from models import *

# train_data = load_train_data()

class Main:
    
    def __init__(self):
        
        # print(train_data)
        pp = preprocessing_data()
        train_data = pp.drop_col('Alley')
        train_data = pp.drop_col('PoolQC')
        train_data = pp.drop_col('Fence')
        train_data = pp.drop_col('MiscFeature')
        pp.fill_null_values()
        train_data = pp.label_encode()
        
        pp.top_k_hot_encode()
        pp.log_transform()
        # print(train_data.columns)
        pp.one_hot_encode()
        print(train_data.iloc[:10,4:8])
        print(train_data)
        nulls = pp.data_info()
        for n in nulls:
            if n !=0:
                print(n)
        # print(nulls)

if __name__ == "__main__":
    Main()
    