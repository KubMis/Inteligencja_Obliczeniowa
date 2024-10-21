import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

if __name__== '__main__':
    df = pd.read_csv("iris.csv")
    numerical_data=df[['sepal.length','sepal.width','petal.length','petal.width']].values
    types_of_irises=df['variety'].values

    (train_set, test_set,train_types,test_types) = train_test_split(numerical_data,types_of_irises, train_size=0.7,
    random_state=300666)

    print(f"train set : \n{train_set}")
    print(f"test set : \n{test_set}")

    model=tree.DecisionTreeClassifier()
    model.fit(train_set, train_types)
    value=model.score(test_set, test_types)

    predictions=model.predict(test_set)
    cnf_matrix=confusion_matrix(test_types,predictions)

    print(f"Confusion matrix is \n {cnf_matrix}")
    print(f"Accuracy of prediction is {value*100} %")

    tree.plot_tree(model, filled=True,
                   feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
                   class_names=model.classes_)
    plt.show()