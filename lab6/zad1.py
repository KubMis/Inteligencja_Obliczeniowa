import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv('titanic.csv')

transactions = df[['Class', 'Sex', 'Age', 'Survived']].values.tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_transformed = pd.DataFrame(te_ary, columns=te.columns_)

print(df_transformed)

frequent_itemsets = apriori(df_transformed, min_support=0.005, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.8, num_itemsets=10)
rules = rules[rules['consequents'].apply(lambda x: 'Yes' in x)]
survival_rules = rules[['antecedents', 'consequents', 'confidence']].sort_values(by='confidence', ascending=False)
print(survival_rules)

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
