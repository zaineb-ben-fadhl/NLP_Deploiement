import pandas as pd
from datasets import load_dataset

ds = load_dataset("andreagasparini/dreaddit")

train_df = pd.DataFrame(ds["train"])
test_df  = pd.DataFrame(ds["test"])

train_df.to_csv("data/dreaddit_train.csv", index=False)
test_df.to_csv("data/dreaddit_test.csv", index=False)
