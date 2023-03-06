import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

print(train_data.columns())
