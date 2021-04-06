
import pandas as pd



def load_train_data():
    
    train_data = pd.read_csv('train.csv')
    return train_data

def load_test_data():
    test_data = pd.read_csv('test.csv')
    return test_data