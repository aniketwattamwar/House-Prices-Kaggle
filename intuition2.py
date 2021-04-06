from utility import *
import pandas as pd
import numpy as np
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


train_data = load_train_data()
test_data = load_test_data()
test_nulls = test_data.isna().sum()
means = train_data.mean()


lotshape_map = {'Reg':4,'IR1':3,'IR2':2,'IR3':1}
land_contour = {'Low':4,'HLS':3,'Bnk':2,'Lvl':1}
land_slope = {'Gtl':1,'Mod':2,'Sev':3}
condition1 = {'Artery':9,'Feedr':8,'Norm':7,'RRNn':6,'RRAn':5,'PosN':4,'PosA':3,'RRNe':2,'RRAe':1}
condition2 = {'Artery':9,'Feedr':8,'Norm':7,'RRNn':6,'RRAn':5,'PosN':4,'PosA':3,'RRNe':2,'RRAe':1}
#bldgType = LabelEncoder
exter_qual = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0}

exter_cond = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0}
bsmt_qual = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,np.nan:0}
bsmt_cond = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,np.nan:0}
bsmt_exposure = {'Gd':4,'Av':3,'Mn':2,'No':1,np.nan:0}
bsmt_finType1 = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,np.nan:0}

bsmt_finType2 = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,np.nan:0}

heating_QC = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0 }

electrical  = {'Mix':0,'FuseP':1,'FuseF':2,'FuseA':3,'SBrkr':4}
kitchen_qual = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0 }

functional = {'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0}

fireplace_qu = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,np.nan:0}
# garage_type =  {'2Types':6,'Attchd':5,'Basement':4,'BuiltIn':3,'CarPort':2,'Detchd':1,np.nan:0}
garage_finish = {'Fin':3,'RFn':2,'Unf':1,np.nan:0 }
garage_qual = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,np.nan:0}
garage_cond = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,np.nan:0}
paved = {'Y':3,'P':2,'N':1}


def encoding(train_data):
    train_data['BldgType'] = label_encoder.fit_transform(train_data['BldgType'])
    train_data['HouseStyle'] = label_encoder.fit_transform(train_data['HouseStyle'])
    train_data['GarageType'] = train_data['GarageType'].fillna(train_data['GarageType'].mode()[0])
    train_data['GarageType'] = label_encoder.fit_transform(train_data['GarageType'])
    train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())
    train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].mean())

    train_data.LotShape = train_data.LotShape.replace(lotshape_map)
    train_data.LandContour = train_data.LandContour.replace(land_contour)
    train_data.LandSlope = train_data.LandSlope.replace(land_slope)
    train_data.Condition1 = train_data.Condition1.replace(condition1)
    train_data.Condition2 = train_data.Condition2.replace(condition2)
    train_data.ExterQual = train_data.ExterQual.replace(exter_qual)
    train_data.ExterCond = train_data.ExterCond.replace(exter_cond)
    train_data.BsmtQual = train_data.BsmtQual.replace(bsmt_qual)
    train_data.BsmtCond = train_data.BsmtCond.replace(bsmt_cond)
    train_data.BsmtExposure = train_data.BsmtExposure.replace(bsmt_exposure)
    train_data.BsmtFinType1 = train_data.BsmtFinType1.replace(bsmt_finType1)
    train_data.BsmtFinType2 = train_data.BsmtFinType2.replace(bsmt_finType2)
    train_data.HeatingQC = train_data.HeatingQC.replace(heating_QC)
    train_data.Electrical = train_data.Electrical.replace(electrical)
    train_data.KitchenQual = train_data.KitchenQual.replace(kitchen_qual)
    train_data.Functional = train_data.Functional.replace(functional)
    train_data.FireplaceQu = train_data.FireplaceQu.replace(fireplace_qu)
    # train_data.GarageType = train_data.GarageType.replace(garage_type)
    train_data.GarageFinish= train_data.GarageFinish.replace(garage_finish)
    train_data.GarageQual = train_data.GarageQual.replace(garage_qual)
    train_data.GarageCond = train_data.GarageCond.replace(garage_cond)
    train_data.PavedDrive = train_data.PavedDrive.replace(paved)
    
    train_data.drop(['Alley','PoolQC','Fence','MiscFeature'], axis = 1,inplace = True)
    
    
    top_neighbors = train_data.Neighborhood.value_counts().sort_values(ascending=False).head(10).index
    top_ext1 = train_data.Exterior1st.value_counts().sort_values(ascending=False).head(6).index
    top_ext2 = train_data.Exterior2nd.value_counts().sort_values(ascending=False).head(6).index
        
    for n in top_neighbors:
        train_data['neighborhood_'+ n] = np.where(train_data['Neighborhood']== n, 1, 0)
    
    for n in top_ext1:
        train_data['top_ext1_'+ n] = np.where(train_data['Exterior1st']== n, 1, 0)
    
    for n in top_ext2:
        train_data['top_ext2_'+ n] = np.where(train_data['Exterior2nd']== n, 1, 0)
    train_data = train_data.drop(['Neighborhood'],axis=1)
    train_data = train_data.drop(['Exterior1st'],axis=1)
    train_data = train_data.drop(['Exterior2nd'],axis=1)
    
    
    # cols = ['MasVnrType','Electrical']
    # for col in cols:
    #     train_data[col] = train_data[col].fillna(train_data[col].mean())
        
    train_data['MasVnrType'] = train_data['MasVnrType'].fillna(train_data['MasVnrType'].mode()[0])
    train_data['Electrical'] = train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])
    train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(train_data['GarageYrBlt'].mode()[0])
    
    return train_data


train_data2 = encoding(train_data)
test_data2 = encoding(test_data)
for column in test_data2.columns:
    test_data2[column] = test_data2[column].fillna(test_data2[column].mode()[0])

nulls_test = test_data2.isna().sum()
nulls = train_data2.isna().sum()


#object data only
obj_data = train_data2.select_dtypes(include=['object'])
obj_data_counts = obj_data.nunique()
#numerical data only
train_data3 = train_data2.select_dtypes(exclude=['object'])

train_en = pd.get_dummies(obj_data)
dfs = [train_en,train_data3]


transformed = train_data3['SalePrice']
train_data3['SalePrice'] = np.log(transformed)
y = train_data3['SalePrice']
train_data3 = train_data3.drop(['SalePrice'],axis =1)

#testing data
test_obj_data = test_data2.select_dtypes(include=['object'])
test_data3 = test_data2.select_dtypes(exclude=['object'])
test_object_counts = test_obj_data.nunique()

test_en = pd.get_dummies(test_obj_data)
test_dfs = [test_en,test_data3]

#align the encoded data
train_en, test_en = train_en.align(test_en, join='inner', axis=1)

training = pd.concat([train_en,train_data3],axis=1)
testing = pd.concat([test_en,test_data3],axis=1)
training_arr = training.iloc[:,:].values
testing_arr = testing.iloc[:,:].values



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(training_arr, y)
regressor.score(training_arr,y)

reg_pred = regressor.predict(testing_arr)
reg_pred = np.exp(reg_pred)
reg_pred = pd.DataFrame(reg_pred)

reg_pred.to_csv('prediction.csv')


import xgboost as xg
xgb_r = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 10, seed = 123)
xgb_r.fit(training_arr, y)
xg_pred = xgb_r.predict(testing_arr)
xg_pred = np.exp(xg_pred)
xg_pred = pd.DataFrame(xg_pred)
xg_pred.to_csv('xg_prediction.csv')

from sklearn.linear_model import ElasticNet
el = ElasticNet(random_state=0)
el.fit(training_arr, y)

el_pred = el.predict(testing_arr)
el_pred = np.exp(el_pred);
el_pred = pd.DataFrame(el_pred)
el_pred.to_csv('el_prediction.csv')


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=0)
rf.fit(training_arr,y)

rf_pred = rf.predict(testing_arr)
rf_pred = np.exp(rf_pred)
rf_pred = pd.DataFrame(rf_pred)
rf_pred.to_csv('rf_prediction.csv')



