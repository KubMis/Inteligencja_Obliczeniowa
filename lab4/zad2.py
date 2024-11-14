import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

if __name__== '__main__':
    df = pd.read_csv("iris.csv")
    numerical_data = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
    types_of_irises = df['variety'].values
    train_set, test_set, train_types, test_types = train_test_split(
        numerical_data,
        types_of_irises,
        train_size=0.7,
        random_state=300666
    )

    classifier=MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(2,),
                    random_state=1)

    classifier.fit(train_set,train_types)
    print(classifier.score(test_set,test_types))

    classifier_3 = MLPClassifier(solver='lbfgs',
                               alpha=1e-5,
                               hidden_layer_sizes=(3,),
                               random_state=1)

    classifier_3.fit(train_set, train_types)
    print(classifier_3.score(test_set, test_types))

    classifier_3_3 = MLPClassifier(solver='lbfgs',
                                 alpha=1e-5,
                                 hidden_layer_sizes=(3,3,),
                                 random_state=1)

    classifier_3_3.fit(train_set, train_types)
    print(classifier_3_3.score(test_set, test_types))

    #architektura z jedną ukrytą warstwą z dwoma neuronami i architektura z dwiema warstwami neuronowymi, po 3 neurony każda

