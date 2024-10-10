import pandas as pd
import numpy as np

dataset = pd.read_csv("iris_with_errors.csv")
missing_values = ["n/a", "NA", "-"]

