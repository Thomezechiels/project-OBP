import vowpalwabbit
import pandas as pd
from pathlib import Path

filepath_train = Path('data/training.csv')  
filepath_train.parent.mkdir(parents=True, exist_ok=True) 
train_df = pd.read_csv(filepath_train)

filepath_test = Path('data/test.csv')  
filepath_test.parent.mkdir(parents=True, exist_ok=True) 
test_df = pd.read_csv(filepath_test)

vw = vowpalwabbit.Workspace("--cb 10", quiet=True)

for i in train_df.index:
    action = train_df.loc[i, "action"]
    cost = train_df.loc[i, "cost"]
    probability = train_df.loc[i, "probability"]
    feature1 = train_df.loc[i, "feature1"]
    feature2 = train_df.loc[i, "feature2"]
    # feature3 = train_df.loc[i, "feature3"]

    # Construct the example in the required vw format.
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
        # + " "
        # + str(feature3)
    )

    # Here we do the actual learning.
    vw.learn(learn_example)

for j in test_df.index:
    feature1 = test_df.loc[j, "feature1"]
    feature2 = test_df.loc[j, "feature2"]
    # feature3 = test_df.loc[j, "feature3"]

    test_example = "| " + str(feature1) + " " + str(feature2)

    choice = vw.predict(test_example)
    print(j, choice)
