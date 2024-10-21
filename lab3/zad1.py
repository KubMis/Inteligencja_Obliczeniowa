import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=300666)

def classify_iris( pl, pw):
    if pw < 1:
        return "Setosa"
    elif pl >= 4.9:
        return "Virginica"
    else:
        return "Versicolor"

good_predictions = 0
length = test_set.shape[0]

for i in range(length):
    if classify_iris(test_set[i, 2],test_set[i, 3]) == test_set[i,4] :
        good_predictions = good_predictions + 1

print(good_predictions)
print(good_predictions / length * 100, "%")