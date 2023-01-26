import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


class RandomForest:
    def __init__(self):
        self.model = None
        self.data = pd.DataFrame(columns = ['num_servers', 'X_t','profit'])
        self.iteration = 0

    def evaluate(self, X_t):        
        max_profit = 0
        optimal_servers = 0
        for num_server in range(10):
            temp_df = pd.DataFrame([[num_server, X_t]],columns = ['num_servers', 'X_t'])
            temp_df[['x1','x2','x3','x4','x5','x6','x7','x8']] = pd.DataFrame(temp_df.X_t.tolist(), index = temp_df.index)
            temp_df = temp_df.drop(['X_t'],axis=1)
            profit = self.model.predict(temp_df)
            print('is de profit ooit positief',profit)
            if profit > max_profit:
                max_profit = profit
                optimal_servers = num_server
        print('profit', profit)
        print('optimal_servers', optimal_servers)
        return optimal_servers
        
    
    def train(self, num_servers, X_t, profit):
        print('training iteration:', self.iteration)
        self.iteration +=1
    
        temp_df = pd.DataFrame([[num_servers, X_t, profit]],columns = ['num_servers', 'X_t','profit'])
        self.data = pd.concat([self.data, temp_df])
        df = self.data
        df[['x1','x2','x3','x4','x5','x6','x7','x8']] = pd.DataFrame(df.X_t.tolist(), index = df.index)
        df = df.drop(['X_t'],axis=1)
        
        target_column = ['profit'] 
        predictors = list(set(list(df.columns))-set(target_column))


        X = df[predictors].values
        y = df[target_column].values
        y=y.astype('int')
 
        
