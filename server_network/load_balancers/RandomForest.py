import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification


class RandomForest:
    def __init__(self):
        self.model = None
        self.data = pd.DataFrame(columns = ['num_servers', 'X_t','profit'])
        self.iteration = 0
        self.history = pd.DataFrame(columns=['max_profit','optimal_servers'])

    def evaluate(self, X_t):        
        max_profit = -1000000000
        optimal_servers = 1
        for num_server in range(1,10):
            temp_df = pd.DataFrame([[num_server, X_t]],columns = ['num_servers', 'X_t'])
            temp_df[['x1','x2','x3','x4']] = pd.DataFrame(temp_df.X_t.tolist(), index = temp_df.index)
            temp_df = temp_df.drop(['X_t'],axis=1)
            profit = self.model.predict(temp_df)
            print('num_server',num_server, 'profit', profit)
            # if num_server == 1:
            #     max_profit = profit
            if profit > max_profit:
                max_profit = profit
                optimal_servers = num_server
        # print('max_profit', max_profit)
        print('optimal_servers', optimal_servers)
        history_df = pd.DataFrame([[max_profit, optimal_servers]],columns = ['max_profit', 'optimal_servers'])
        self.history = pd.concat([self.history, history_df])
        return optimal_servers
        
    
    def train(self, num_servers, X_t, profit):
        # print('training iteration:', self.iteration)
        self.iteration +=1

        # print(num_servers, X_t, profit);

        temp_df = pd.DataFrame([[num_servers, X_t, profit]],columns = ['num_servers', 'X_t','profit'])
        self.data = pd.concat([self.data, temp_df])
        df = self.data
        df[['x1','x2','x3','x4']] = pd.DataFrame(df.X_t.tolist(), index = df.index)
        df = df.drop(['X_t'],axis=1)
        
        target_column = ['profit'] 
        predictors = list(set(list(df.columns))-set(target_column))


        X = df[predictors].values
        y = df[target_column].values
        y=y.astype('int')

        self.model = RandomForestRegressor(n_estimators=50, random_state=44)
        self.model.fit(X, y)