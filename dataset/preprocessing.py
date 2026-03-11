import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


food=pd.read_csv("food_dataset.csv")

print(food.shape)
print(food.info)
print(food.head())
print(food.columns)
print(food.isnull().sum())
print(food.describe())
print(food["meal_category"].value_counts())

sns.boxplot(x=food["Prices"])
sns.boxplot(x=food["calories"])
sns.boxplot(x=food["protein"])
sns.boxplot(x=food["carbs"])
sns.boxplot(x=food["fat"])
sns.boxplot(x=food["nutrition_score"])

plt.show()

upper = food["Prices"].quantile(0.95)
lower = food["Prices"].quantile(0.05)
food["Prices"] = np.clip(food["Prices"], lower, upper)

print(food.nunique())
print(food.duplicated().sum())
print(food[food.duplicated()])
food = food.drop_duplicates()
food = food.drop(columns=["name","course"])
