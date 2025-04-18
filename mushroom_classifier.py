import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("mushrooms.csv")
X = pd.get_dummies(df.drop('class', axis=1))
y = df['class']

# Split into training and testing sets
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

# Convert to DataFrame and print results
results_df = pd.DataFrame(results)
print(results_df)

# Visualization - Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
results_df.set_index('Model').plot(kind='bar', ax=ax)
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot in the Results folder
plt.savefig('Results/model_performance_comparison.png')

# Show the plot
plt.show()
