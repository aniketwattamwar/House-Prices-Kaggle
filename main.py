

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
        print(train_data.columns)
        
        


if __name__ == "__main__":
    Main()