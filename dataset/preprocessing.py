import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


food=pd.read_csv("/Users/syedadnan/Downloads/Internship/Mini project /Mini Project 2/nutritional_accessibility_classification/dataset/food_dataset.csv")

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
print(food.corr(numeric_only=True))
sns.heatmap(food.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()
print(food.head())
print(food.info)
print(food.describe())

X = food.drop("meal_category", axis=1)
y = food["meal_category"]

X.to_csv("features_X.csv", index=False)
y.to_csv("target_y.csv", index=False)