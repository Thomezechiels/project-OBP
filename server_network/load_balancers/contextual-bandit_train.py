import vowpalwabbit
import pandas as pd
from pathlib import Path

filepath_train = Path('data/training.csv')  
filepath_train.parent.mkdir(parents=True, exist_ok=True) 
train_df = pd.read_csv(filepath_train)

vw = vowpalwabbit.Workspace("--cb_explore 10 --cover 8", quiet=True)

train_df = train_df.sort_values(by=['feature_arrival', 'feature_workload', 'action', 'cost'])
train_df = train_df.groupby(['action', 'feature_arrival', 'feature_workload']).tail(3)

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

