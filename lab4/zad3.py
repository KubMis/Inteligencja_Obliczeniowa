import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

if __name__== '__main__':
    df = pd.read_csv('diabetes.csv')
    numerical_data = df[['pregnant-times','glucose-concentr','blood-pressure','skin-thickness','insulin','mass-index','pedigree-func','age']].values
    results = df['class'].values
    train_set, test_set, train_results, test_results = train_test_split(
numerical_data,
        results,
        train_size=0.7,
        random_state=300666
    )

    classifier_neuron = MLPClassifier(solver='lbfgs',
                                      alpha=1e-5,
                                      activation='relu',
                                      hidden_layer_sizes=(6,3,),
                                      random_state=1,
                                      max_iter=499)

    classifier_neuron.fit(train_set, train_results)
    score = classifier_neuron.score(test_set, test_results)
    classifier_predictions=classifier_neuron.predict(test_set)

    classifier_neuron_different = MLPClassifier(solver='lbfgs',
                                      alpha=1e-5,
                                      activation='tanh',
                                      hidden_layer_sizes=(8, 4,),
                                      random_state=1,
                                      max_iter=499)

    classifier_neuron_different.fit(train_set, train_results)
    score2 = classifier_neuron_different.score(test_set, test_results)

    decision_tree_classifier=tree.DecisionTreeClassifier().fit(train_set, train_results)
    score3 = decision_tree_classifier.score(test_set, test_results)

    knn5_classifier=KNeighborsClassifier(n_neighbors=5).fit(train_set, train_results)
    score4 = knn5_classifier.score(test_set, test_results)

    naive_bayes_classifier = GaussianNB().fit(train_set, train_results)
    score5=naive_bayes_classifier.score(test_set, test_results)

    cnf_matrix= confusion_matrix(test_results, classifier_predictions)
    print(f"neuron 6,3 score is {score}")
    print(cnf_matrix)
    print(f"neuron net 8,4 score is {score2}")
    print(f"decision tree score is {score3}")
    print(f"Knn5  score is {score4}")
    print(f"Naive bayes score is {score5}")