import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

if __name__== '__main__':
    df = pd.read_csv("iris.csv")
    numerical_data=df[['sepal.length','sepal.width','petal.length','petal.width']].values
    types_of_irises=df['variety'].values

    (train_set, test_set,train_types,test_types) = train_test_split(numerical_data,types_of_irises, train_size=0.7,
    random_state=300666)

    naive_bayes=GaussianNB().fit(train_set, train_types)
    knn_3 = KNeighborsClassifier(n_neighbors=3).fit(train_set, train_types)
    knn_5 = KNeighborsClassifier(n_neighbors=5).fit(train_set, train_types)
    knn_11 = KNeighborsClassifier(n_neighbors=11).fit(train_set, train_types)

    nb_predictions = naive_bayes.predict(test_set)
    nb_cnf_matrix = confusion_matrix(test_types, nb_predictions)

    knn_3_predictions = knn_3.predict(test_set)
    knn_3_cnf_matrix = confusion_matrix(test_types, knn_3_predictions)

    knn_5_predictions = knn_5.predict(test_set)
    knn_5_cnf_matrix = confusion_matrix(test_types, knn_5_predictions)

    knn_11_predictions = knn_11.predict(test_set)
    knn_11_cnf_matrix = confusion_matrix(test_types, knn_11_predictions)

    print(f"Naive Bayes Gausian score = {naive_bayes.score(test_set,test_types)}")
    print(f"Confusion matrix for Naive Bayes:\n{nb_cnf_matrix}")
    print(f"3 nearest neighbours score = {knn_3.score(test_set,test_types)}")
    print(f"Confusion matrix for 3 nearest neighbours:\n{knn_3_cnf_matrix}")
    print(f"5 nearest neighbours score = {knn_5.score(test_set, test_types)}")
    print(f"Confusion matrix for 5 nearest neighbours:\n{knn_5_cnf_matrix}")
    print(f"11 nearest neighbours score = {knn_11.score(test_set, test_types)}")
    print(f"Confusion matrix for 11 nearest neighbours:\n{knn_11_cnf_matrix}")
