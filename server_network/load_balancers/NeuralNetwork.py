import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import numpy
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(8,8,8,8,8), activation='relu', solver='adam', max_iter=500)
        self.data = pd.DataFrame(columns = ['num_servers', 'X_t','profit'])
        self.iteration = 0
        self.last_profit = False
        self.differences = []
        self.accuracy_df = pd.DataFrame(columns=['iteration','accuracy'])
        self.current_profit_period = 0
        self.profit_df = pd.DataFrame(columns=['iteration','profit'])
        self.temp_itteration = 0

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
                self.last_profit = profit
        # print('max_profit', max_profit)
        # print('optimal_servers', optimal_servers)
        return optimal_servers
        
    
    def train(self, num_servers, X_t, profit):
        print('training iteration:', self.iteration)
        self.iteration +=1

        if (self.last_profit):
            difference = (abs(profit - self.last_profit[0])/abs(profit)) * 100
            #difference = abs(profit - self.last_profit[0])
            if difference < 100:
                self.differences.append(difference)
                if (len(self.differences) > 24):
                    self.differences.pop(0)
                total_difference = round(sum(self.differences) / len(self.differences), 2)
                acc_df = pd.DataFrame([[self.iteration, total_difference]],columns = ['iteration','error'])
                self.accuracy_df = pd.concat([self.accuracy_df, acc_df])
                print('Difference_moving:', total_difference)
                print('Difference:', difference)
                print('Profit:', profit, self.last_profit[0])
            self.last_profit = False

            self.current_profit_period += profit
            self.temp_itteration += 1

            if self.iteration % 12 == 0 and self.iteration > 13:
                prof_df = pd.DataFrame([[self.iteration, (self.current_profit_period/self.temp_itteration)]],columns = ['iteration','profit'])
                self.profit_df = pd.concat([self.profit_df, prof_df])
                self.current_profit_period = 0
                self.temp_itteration = 0
            elif self.iteration % 12 == 0  and self.iteration < 13:
                self.current_profit_period = 0
                self.temp_itteration = 0

        if (self.iteration == 998):
            self.profit_df.to_csv("neural_network.csv")
            # self.accuracy_df.plot(x='iteration', y="error")
            # plt.ylabel("Error percentage")
            # plt.xlabel("Iteration")
            # plt.grid()
            # plt.title("Neural network moving error")
            # plt.show()

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

        self.model.fit(X,y)