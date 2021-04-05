

from utility import *
import numpy as np
train_data = load_train_data()


def replace_labels(col_name,d):
    train_data[col_name] = train_data[col_name].replace(d)


class preprocessing_data:
    
    def __init__(self):
        self.train_data = train_data
        
        
    
    def fill_null_values(self):
        
        self.train_data = self.train_data.fillna(self.train_data['LotFrontage'].mean())

        return self.train_data
    
    def one_hot_encode(self):
        
        # train_data['GarageFinish'] = pd.get_dummies(train_data.GarageFinish, prefix='GarageFinish_',drop_first=True)
        
        
        
        print(self.train_data)
        return 0
    
    def label_encode(self):
        
        #LotShape
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
        
        # garage_type =  {'2Types':6,'Attchd':5,'Basement':4,'BuiltIn':3,'CarPort':2,'Detchd':1,'NA':0}
        garage_qual = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,np.nan:0}
        garage_cond = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,np.nan:0}
        paved = {'Y':3,'P':2,'N':1}
        
        
        # self.train_data['BsmtCond'] = self.train_data['BsmtCond'].replace(bsmt_cond)
        self.train_data.LotShape = self.train_data.LotShape.replace(lotshape_map)
        self.train_data.LandContour = self.train_data.LandContour.replace(land_contour)
        self.train_data.LandSlope = self.train_data.LandSlope.replace(land_slope)
        self.train_data.Condition1 = self.train_data.Condition1.replace(condition1)
        self.train_data.Condition2 = self.train_data.Condition2.replace(condition2)
        self.train_data.ExterQual = self.train_data.ExterQual.replace(exter_qual)
        self.train_data.ExterCond = self.train_data.ExterCond.replace(exter_cond)
        self.train_data.BsmtQual = self.train_data.BsmtQual.replace(bsmt_qual)
        self.train_data.BsmtCond = self.train_data.BsmtCond.replace(bsmt_cond)
        self.train_data.BsmtExposure = self.train_data.BsmtExposure.replace(bsmt_exposure)
        self.train_data.BsmtFinType1 = self.train_data.BsmtFinType1.replace(bsmt_finType1)
        self.train_data.BsmtFinType2 = self.train_data.BsmtFinType2.replace(bsmt_finType2)
        self.train_data.HeatingQC = self.train_data.HeatingQC.replace(heating_QC)
        self.train_data.Electrical = self.train_data.Electrical.replace(electrical)
        self.train_data.KitchenQual = self.train_data.KitchenQual.replace(kitchen_qual)
        self.train_data.Functional = self.train_data.Functional.replace(functional)
        self.train_data.FireplaceQu = self.train_data.FireplaceQu.replace(fireplace_qu)
        self.train_data.GarageQual = self.train_data.GarageQual.replace(garage_qual)
        self.train_data.GarageCond = self.train_data.GarageCond.replace(garage_cond)
        self.train_data.PavedDrive = self.train_data.PavedDrive.replace(paved)
            
        
        return self.train_data
    
    def top_k_hot_encode(self):
        
        top_neighbors = self.train_data.Neighborhood.value_counts().sort_values(ascending=False).head(10).index
        top_ext1 = self.train_data.Exterior1st.value_counts().sort_values(ascending=False).head(6).index
        top_ext2 = self.train_data.Exterior2nd.value_counts().sort_values(ascending=False).head(6).index
            
        for n in top_neighbors:
            self.train_data['neighborhood_'+ n] = np.where(self.train_data['Neighborhood']== n, 1, 0)
        
        for n in top_ext1:
            self.train_data['top_ext1_'+ n] = np.where(self.train_data['Exterior1st']== n, 1, 0)
        
        for n in top_ext2:
            self.train_data['top_ext2_'+ n] = np.where(self.train_data['Exterior2nd']== n, 1, 0)
        self.train_data = self.train_data.drop(['Neighborhood'],axis=1)
        self.train_data = self.train_data.drop(['Exterior1st'],axis=1)
        self.train_data = self.train_data.drop(['Exterior2nd'],axis=1)
        return 0
    
    def log_transform(self):
        
        transformed = self.train_data['SalePrice']
        self.train_data['SalePrice'] = np.log(transformed)
         
        return 0
    
    def drop_col(self,col_name):
        print(col_name)
        self.train_data = self.train_data.drop([col_name],axis=1)
        return self.train_data
    
    def data_info(self):
        print(self.train_data.info())
        print(self.train_data.columns)
        print(self.train_data['SalePrice'])
        nulls = self.train_data.isna().sum()        
        return nulls
    

 
    
    
    

