import pandas as pd
from sklearn.neural_network import MLPRegressor
import numpy
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self):
        self.model = None
        self.data = pd.DataFrame(columns = ['num_servers', 'X_t','profit'])
        self.iteration = 0

    def evaluate(self, X_t):        
        max_profit = -1000000
        optimal_servers = 1
        for num_server in range(1,10):
            temp_df = pd.DataFrame([[num_server, X_t]],columns = ['num_servers', 'X_t'])
            temp_df[['x1','x2','x3','x4']] = pd.DataFrame(temp_df.X_t.tolist(), index = temp_df.index)
            temp_df = temp_df.drop(['X_t'],axis=1)

            profit = self.model.predict(temp_df)
            if profit > max_profit:
                max_profit = profit
                optimal_servers = num_server
        print('max_profit', max_profit)
        print('optimal_servers', optimal_servers)
        return optimal_servers
        
    
    def train(self, num_servers, X_t, profit):
        print('training iteration:', self.iteration)
        print('training profit:', profit)
        self.iteration +=1

        temp_df = pd.DataFrame([[num_servers, X_t, profit]],columns = ['num_servers', 'X_t','profit'])
        self.data = pd.concat([self.data, temp_df])
        df = self.data
        df[['x1','x2','x3','x4']] = pd.DataFrame(df.X_t.tolist(), index = df.index)
        df = df.drop(['X_t'],axis=1)

        X = df.drop(['profit'],axis=1)
        
        target_column = ['profit'] 

        #predictors = list(set(list(df.columns))-set(target_column))
        # X = df[predictors]
        
        y = df[target_column]
        y=y.astype('int')

        #train/test split nog implementeren?
        # if self.iteration>10:
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
 
        self.model = MLPRegressor(hidden_layer_sizes=(8,8,8,8,8), activation='relu', solver='adam', max_iter=500)
        self.model.fit(X,y)