import numpy as np
from scipy.stats import zscore
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()
FLOWER_TYPE = pd.Series(iris.target, name='FlowerType')
def create_plot_all_plots():
    sepal_length_data = pd.DataFrame(iris.data[:, 0], columns=['sepal length (cm)'])
    sepal_width_data = pd.DataFrame(iris.data[:, 1], columns=['sepal width (cm)'])

    normalized_length = sepal_length_data/sepal_length_data.abs().max()
    normalized_width = sepal_width_data/sepal_width_data.abs().max()

    zscored_length= sepal_length_data.apply(zscore)
    zscored_width = sepal_width_data.apply(zscore)

    create_single_plot(sepal_length_data,sepal_width_data,"Original")
    create_single_plot(normalized_length, normalized_width, "Normalized")
    create_single_plot(zscored_length,zscored_width,"Zscored")

def create_single_plot(x,y,title):
    for i in np.unique(FLOWER_TYPE):
        ix = np.where(FLOWER_TYPE == i)
        plt.scatter(x.iloc[ix], y.iloc[ix], label=iris.target_names[i])

    plt.title(title)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend()
    plt.show()

create_plot_all_plots()