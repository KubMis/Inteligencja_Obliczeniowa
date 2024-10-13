import pandas as pd

missing_values = ["NA", "-", " "]
numerical_column_names = ["sepal.length","sepal.width","petal.length","petal.width"]
iris_variety = ["Setosa","Versicolor", "Virginica"]
dataset = pd.read_csv("iris_with_errors.csv", na_values=missing_values)

print(dataset.isnull().sum())

def correct_na_values():
    for names in numerical_column_names:
        na_count = dataset[names].isna().sum()
        if na_count > 0:
            print(f"Changing {na_count} values NA in column {names}")
        dataset[names] = dataset[names].fillna(dataset[names].median(skipna=True))

def correct_numerical_values():
    for names in numerical_column_names :
        for rows in dataset[names] :
            if 0>rows or rows>=15 :
                print(f"Changing {rows} in {names} to ")
                rows=dataset[names].median()
                print(f"{rows}")

def get_first_3_chars(name_from_file):
    for name in iris_variety:
        if name_from_file[0:3].lower() == name[0:3].lower() :
            name_from_file = name
            return name_from_file

def correct_iris_names():
    for names in dataset['variety']:
        if names not in iris_variety:
            changed_name = get_first_3_chars(names)
            print(f" changing {names} to {changed_name}")

correct_na_values()
correct_numerical_values()
correct_iris_names()
