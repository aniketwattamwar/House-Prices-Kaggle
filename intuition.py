
from utility import *

train_data = load_train_data()
import numpy as np

null_values = train_data.isna().sum()
GarageType = train_data['GarageType']

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
HouseStyle = label_encoder.fit_transform(train_data['HouseStyle'])


# from sklearn.preprocessing import OrdinalEncoder
# enc = OrdinalEncoder()
# ext = enc.fit(train_data['ExterQual'])



lotshape_map = {'Reg':4,'IR1':3,'IR2':2,'IR3':1}
land_contour = {'Low':4,'HLS':3,'Bnk':2,'Lvl':1}
lan_slope = {'Gtl':1,'Mod':2,'Sev':3}
condition1 = {'Artery':9,'Feedr':8,'Norm':7,'RRNn':6,'RRAn':5,'PosN':4,'PosA':3,'RRNe':2,'RRAe':1
              }
condition2 = {'Artery':9,'Feedr':8,'Norm':7,'RRNn':6,'RRAn':5,'PosN':4,'PosA':3,'RRNe':2,'RRAe':1}

#bldgType = LabelEncoder
exter_qual = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0}

exter_cond = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0}
bsmt_qual = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,np.nan:0}
bsmt_cond = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,np.nan:0}
bsmt_exposure = {'Gd':4,'Av':3,'Mn':2,'No':1,'NA':0}
bsmt_finType1 = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0}

bsmt_finType2 = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0}

heating_QC = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0 }

electrical  = {'Mix':0,'FuseP':1,'FuseF':2,'FuseA':3,'SBrkr':4}
kitchen_qual = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0 }

functional = {'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0}

fireplace_qu = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}

# garage_type =  {'2Types':6,'Attchd':5,'Basement':4,'BuiltIn':3,'CarPort':2,'Detchd':1,'NA':0}
garage_qual = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
garag_cond = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
paved = {'Y':3,'P':2,'N':1}


# train_data.LotShape = train_data.LotShape.map(lotshape_map)

train_data['BsmtCond7'] = train_data['BsmtCond'].replace(bsmt_cond)

bsmt_null = train_data['BsmtCond'].isna().sum()



















