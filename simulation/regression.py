from sklearn import linear_model
import pandas as pd
import math
import warnings

class Regressor:

    def __init__(self):
        self.model = self.trainmodel()

    def trainmodel(self):
        df = pd.read_csv('data/training.csv')
        X = df[['feature1','feature2']]
        y = df['feature3']
        model = linear_model.LinearRegression()
        model.fit(X,y)
        return model

    def determineservers(self,config, arrivalprob):
        prediction = self.model.predict([[arrivalprob,config['prob_small']]])
        return math.ceil(prediction / config['steps'])



       
