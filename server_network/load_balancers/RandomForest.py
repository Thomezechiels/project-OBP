import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib

class RandomForest:
    def __init__(self):
        filename = 'data/models/random_forest.sav'
        self.model = joblib.load(filename)
        self.data = pd.DataFrame(columns = ['num_servers', 'X_t','profit'])

    def evaluate(self, X_t):        
        max_profit = -10000000
        optimal_servers = 1
        for num_server in range(1,10):
            temp_df = pd.DataFrame([[num_server, X_t]],columns = ['num_servers', 'X_t'])
            temp_df[['x1','x2','x3','x4']] = pd.DataFrame(temp_df.X_t.tolist(), index = temp_df.index)
            temp_df = temp_df.drop(['X_t'],axis=1)
            profit = self.model.predict(temp_df)
            if profit > max_profit:
                max_profit = profit
                optimal_servers = num_server
        return optimal_servers
    
    def evaluate_live(self, X_t):        
        ret = {1: {'servers': [], 'profit': []}, 2: {'servers': [], 'profit': []}, 3: {'servers': [], 'profit': []}}
        profits = []
        for num_server in range(1,10):
            temp_df = pd.DataFrame([[num_server, X_t]],columns = ['num_servers', 'X_t'])
            temp_df[['x1','x2','x3','x4']] = pd.DataFrame(temp_df.X_t.tolist(), index = temp_df.index)
            temp_df = temp_df.drop(['X_t'],axis=1)
            profit = self.model.predict(temp_df)
            profits.append(profit[0])
        profits = np.array(profits)
        ind = np.argpartition(profits, -3)[-3:]
        ind = ind[np.argsort(profits[ind])]
        for idx, index in enumerate(ind):
            ret[idx+1]['servers'] = index + 1
            ret[idx+1]['profit'] = profits[index]
        return ret
    
    def train(self, num_servers, X_t, profit):    
        temp_df = pd.DataFrame([[num_servers, X_t, profit]],columns = ['num_servers', 'X_t','profit'])
        self.data = pd.concat([self.data, temp_df])
        df = self.data
        df[['x1','x2','x3','x4']] = pd.DataFrame(df.X_t.tolist(), index = df.index)
        df = df.drop(['X_t'],axis=1)
        
        target_column = ['profit']
        X = df.drop(['profit'],axis=1)
        target_column = ['profit'] 
        y = df[target_column].values.ravel()
        y=y.astype('int')
        self.model.fit(X,y)