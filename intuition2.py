from utility import *
import pandas as pd
import numpy as np
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


train_data = load_train_data()
test_data = load_test_data()
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
garage_type =  {'2Types':6,'Attchd':5,'Basement':4,'BuiltIn':3,'CarPort':2,'Detchd':1,np.nan:0}
garage_finish = {'Fin':3,'RFn':2,'Unf':1,np.nan:0 }
garage_qual = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,np.nan:0}
garage_cond = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,np.nan:0}
paved = {'Y':3,'P':2,'N':1}


def encoding(train_data):
    train_data['BldgType'] = label_encoder.fit_transform(train_data['BldgType'])
    train_data['HouseStyle'] = label_encoder.fit_transform(train_data['HouseStyle'])
    
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
    train_data.GarageType = train_data.GarageType.replace(garage_type)
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

nulls = train_data2.isna().sum()
#object data only
obj_data = train_data2.select_dtypes(include=['object'])

#numerical data only
train_data3 = train_data2.select_dtypes(exclude=['object'])


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(drop='first')
enc_df2 = pd.DataFrame(enc.fit_transform(obj_data.astype(str)).toarray())

dfs = [enc_df2,train_data3]
training = pd.concat([enc_df2,train_data3],axis=1)























