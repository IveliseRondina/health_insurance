import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = self.feature_engineering(X)
        return X
    
    def feature_engineering(self, data):
        data.vehicle_damage = data.vehicle_damage.apply(lambda x:0 if x == 'No' else 1)
        
        data.vehicle_age = data.vehicle_age.apply(lambda x:1 if x == '< 1 Year' else 2 if x == '1-2 Year' else 3)
        
        gender = {'Male': 1, 'Female': 0}
        data.gender = data.gender.map(gender)
        
        data['premium_vintage'] = data.annual_premium/data.vintage
        
        data['saleschannel_region'] = data.policy_sales_channel / (data.region_code + 1)
        
        data['age_vintage'] = data.age / data.vintage
        
        data['premium_age'] = data.annual_premium / data.age
             
        return data