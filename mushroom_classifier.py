import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the full dataset
df = pd.read_csv("mushrooms.csv")

# Randomly sample 10% of the data (around 800 samples if the dataset has 8,124 samples)
df_sampled = df.sample(frac=0.1, random_state=42)

# Separate features (X) and target (y)
X = pd.get_dummies(df_sampled.drop('class', axis=1))  # One-hot encode categorical features
y = df_sampled['class']  # Target variable (class)

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'SVM': SVC()
}

# Evaluate each model
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 2),
        'Precision': round(precision_score(y_test, y_pred, pos_label='p'), 2),
        'Recall': round(recall_score(y_test, y_pred, pos_label='p'), 2),
        'F1-Score': round(f1_score(y_test, y_pred, pos_label='p'), 2)
    })

# Convert results to a DataFrame and print
results_df = pd.DataFrame(results)

# Print class distribution
print("Class Distribution:")
print(df['class'].value_counts())

# Print odor vs class breakdown
print("\nOdor vs Class:")
odor_vs_class = df.groupby(['odor', 'class']).size().unstack().fillna(0)
print(odor_vs_class)

# Print bruises vs class breakdown
print("\nBruises vs Class:")
bruises_vs_class = df.groupby(['bruises', 'class']).size().unstack().fillna(0)
print(bruises_vs_class)

# Print habitat vs class breakdown
print("\nHabitat vs Class:")
habitat_vs_class = df.groupby(['habitat', 'class']).size().unstack().fillna(0)
print(habitat_vs_class)

# Print top 10 most important features (using Random Forest)
rf_model = models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nTop 10 Most Important Features:")
print(importances.sort_values(ascending=False).head(10))

# Print model performance
print("\nModel Performance:")
print(results_df)

# Visualization: Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
results_df.set_index('Model').plot(kind='bar', ax=ax)
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save the plot in the Results folder
plt.savefig('Results/model_performance_comparison.png')
plt.show()
