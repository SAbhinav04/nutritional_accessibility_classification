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
