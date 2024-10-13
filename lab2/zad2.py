from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot  as plt

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
#y = pd.Series(iris.target, name='FlowerType')
print(X.head())


pca_iris_2 = PCA(n_components=2).fit(iris.data)
pca_iris_3 = PCA(n_components=3).fit(iris.data)

'''print(pca_iris_2)
print("procent zachowanych danych")
print(pca_iris_2.explained_variance_ratio_) # dla 2 kolumn  0.92461872 + 0.05306648 = 0.9776852, dla jednej 0.92461872
print(pca_iris_2.components_)
print(pca_iris_2.transform(iris.data))
'''

plt.figure()
x2d = pca_iris_2.transform(iris.data)[:, 0]
y2d= pca_iris_2.transform(iris.data)[:, 1]
plt.scatter(x2d,y2d, c=iris.target)
plt.title('PCA dla 2 kolumn')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x3d = pca_iris_3.transform(iris.data)[:, 0]
y3d= pca_iris_3.transform(iris.data)[:, 1]
z3d=pca_iris_3.transform(iris.data)[:, 2]
ax.scatter(x3d, y3d, z3d, c=iris.target)
ax.set_title('PCA dla 3 kolumn')
plt.show()

