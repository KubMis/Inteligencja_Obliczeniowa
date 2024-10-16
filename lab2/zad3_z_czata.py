import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
'''
twórz wykres punktowy dla danych o irysach Y - sepal width X - sepla length
rodzaje irysów oznacz w legendzie, dodatkowo jeden wykres ma być znormalizowany min ma
x drugi ma być znormalizowany za pomocą zscore, a trzeci ma byc zwykły.
Użyj do tego pythona

'''
# Wczytanie danych o irysach
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Zamiana nazw kolumn na bardziej przystępne
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

# Min-Max normalizacja
scaler_minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[['sepal length', 'sepal width']] = scaler_minmax.fit_transform(df_minmax[['sepal length', 'sepal width']])

# Z-score normalizacja
scaler_zscore = StandardScaler()
df_zscore = df.copy()
df_zscore[['sepal length', 'sepal width']] = scaler_zscore.fit_transform(df_zscore[['sepal length', 'sepal width']])

# Tworzenie wykresu
plt.figure(figsize=(21, 6))

# Zwykły wykres bez normalizacji
plt.subplot(1, 3, 1)
sns.scatterplot(x='sepal length', y='sepal width', hue='species', data=df, palette='deep')
plt.title('Original Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Wykres dla normalizacji Min-Max
plt.subplot(1, 3, 2)
sns.scatterplot(x='sepal length', y='sepal width', hue='species', data=df_minmax, palette='deep')
plt.title('Min-Max Normalization')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Wykres dla normalizacji Z-score
plt.subplot(1, 3, 3)
sns.scatterplot(x='sepal length', y='sepal width', hue='species', data=df_zscore, palette='deep')
plt.title('Z-score Normalization')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Wyświetlanie wykresów
plt.tight_layout()
plt.show()
