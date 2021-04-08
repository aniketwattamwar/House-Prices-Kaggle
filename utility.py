
import pandas as pd
import yaml


def load_train_data():
    
    train_data = pd.read_csv('train.csv')
    return train_data

def load_test_data():
    test_data = pd.read_csv('test.csv')
    return test_data


def load_config():
    with open('config.yaml') as file:
        config = yaml.safe_load(file)
    return config





