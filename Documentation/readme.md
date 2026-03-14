Here’s a **more professional GitHub-style README**. This version is **cleaner, recruiter-friendly, and structured like real ML repos**.

# Nutritional Accessibility Classification

A machine learning project that classifies food items based on nutritional value and affordability using supervised learning algorithms.

This project demonstrates how AI can analyze food data and categorize meals into accessibility groups using nutritional attributes such as price, calories, protein, carbohydrates, fat, and nutrition score.

---

# Project Motivation

Access to nutritious and affordable food is an important global challenge. Many food options vary greatly in nutritional value and price, making it difficult to determine healthy and accessible choices.

This project applies machine learning techniques to classify food items into meaningful nutritional accessibility categories and demonstrate how AI can support food and nutrition analysis.

---

# Dataset

The dataset contains structured food data with nutritional attributes.

### Features

| Feature | Description |
|------|-------------|
| Price | Cost of the food item |
| Calories | Energy content |
| Protein | Protein content |
| Carbohydrates | Carbohydrate content |
| Fat | Fat content |
| Nutrition Score | Overall health score |

### Target Variable

**Meal Category**

Represents the classification of food accessibility such as:

- Affordable Nutritious
- Expensive Nutritious
- Low Nutrition
- Basic Meal

---

# Machine Learning Models

Two supervised learning algorithms were implemented.

## K-Nearest Neighbors (KNN)

KNN is a similarity-based classification algorithm that assigns a category to a new food item based on the majority class among its nearest neighbors.

Distance between data points is calculated using Euclidean distance.

## Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and robustness.

---

# Project Pipeline


Food Dataset
↓
Data Loading
↓
Data Cleaning
↓
Outlier Handling
↓
Exploratory Data Analysis
↓
Feature / Target Split
↓
Train-Test Split (80:20)
↓
Model Training (KNN, Random Forest)
↓
Prediction
↓
Model Evaluation


---

# Data Preprocessing

The following preprocessing steps were applied:

- Removal of duplicate records
- Feature selection
- Outlier handling using percentile clipping
- Exploratory Data Analysis (EDA)
- Train-test split

These steps ensure data quality and improve model reliability.

---

# Evaluation Metrics

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics provide insights into the model’s classification performance.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

# Repository Structure

```

nutritional_accessibility_classification
│
├── dataset
│   └── food_dataset.csv
│
├── models
│   ├── knn_distance.py
│   └── random_forest.py
│
├── evaluation
│   └── evaluation.py
│
├── documentation
│   └── project_report.pdf
│
└── README.md

```

---

# Results

The models successfully learned patterns between nutritional attributes and food accessibility categories.

Evaluation using classification metrics and confusion matrices showed that machine learning can effectively analyze food datasets and categorize food items based on nutritional value and affordability.

---

# Challenges

Key challenges encountered during the project:

- Handling duplicate and inconsistent dataset entries
- Managing outliers in nutritional features
- Selecting optimal K values for the KNN algorithm
- Debugging preprocessing and model training pipelines

---

# Future Improvements

Possible enhancements include:

- Adding more nutritional attributes such as fiber, vitamins, and minerals
- Implementing advanced algorithms like Support Vector Machines or Gradient Boosting
- Hyperparameter tuning and cross-validation
- Building a web application for real-time food classification

---

# Contributors

- Ananya H V
- Jashwanth P
- Pavitra Narayan Bhat
- Prajwal J B
- Deepthi Y
- Syed Mohammed Adnan
- Saniya Naaz
- Punit Murali
- Chandan D U
- Rakshitha S
- S Abhinav
- Prerna Rao
- Shifa Firdose
- Kaviya S
- Sanjitkumar SS
- Mohammadiya Risaldar

---

# License

This project was developed as part of an internship program on Artificial Intelligence and Machine Learning.


