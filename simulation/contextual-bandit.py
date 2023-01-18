import vowpalwabbit
import pandas as pd
from pathlib import Path

filepath_train = Path('data/training.csv')  
filepath_train.parent.mkdir(parents=True, exist_ok=True) 
train_df = pd.read_csv(filepath_train)

# filepath_test = Path('data/test.csv')  
# filepath_test.parent.mkdir(parents=True, exist_ok=True) 
# test_df = pd.read_csv(filepath_test)

vw = vowpalwabbit.Workspace("--cb_explore 10 --cover 8", quiet=True)

# Normalize workload feature
# train_df['feature_workload'] = (train_df['feature_workload'] - train_df['feature_workload'].min()) / (train_df['feature_workload'].max() - train_df['feature_workload'].min())

# print(train_df[(train_df['feature_workload'] == 6000) & (train_df['feature_arrival'] == 0.26)])

train_df = train_df.sort_values(by=['feature_arrival', 'feature_workload', 'action', 'cost'])

# train_df = train_df.groupby(['feature_arrival', 'feature_workload']).tail(3)

train_df = train_df.groupby(['action', 'feature_arrival', 'feature_workload']).tail(3)
# print(train_df[(train_df['feature_arrival'] == 0.1) & (train_df['feature_workload'] == 0)])
# print(train_df[(train_df['feature_arrival'] == 0.05) & (train_df['feature_workload'] == 0)])
# print(train_df)
for i in train_df.index:
    action = train_df.loc[i, "action"]
    cost = train_df.loc[i, "cost"]
    probability = train_df.loc[i, "probability"]
    feature1 = train_df.loc[i, "feature_arrival"]
    feature2 = train_df.loc[i, "feature_workload"]

    learn_example = (
        str(action)
        + ":"
        + str(cost)
        + ":"
        + str(probability)
        + " | "
        + str(feature1)
        + " "
        + str(feature2)
    )
    vw.learn(learn_example)

vw.save("cb.model")

