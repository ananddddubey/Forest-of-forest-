import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
import graphviz

# Load the dataset
df = pd.read_csv(r"C:\Users\jidub\OneDrive\Documents\New folder\car_evaluation.csv")

# Check the first few rows of the dataset
print(df.head())

# Prepare the data for training by encoding categorical variables
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Check the statistical summary of the dataset
print(df.describe())

# Calculate the correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Prepare the data for training
X = df.drop('buying', axis=1)
y = df['buying']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the forest of forests ensemble
forest_of_forests = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

# Train the model
forest_of_forests.fit(X_train, y_train)

# Make predictions
y_pred= forest_of_forests.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot the decision tree structure for the first tree in the first forest
dot_data = graphviz.Source(forest_of_forests.estimators_[0][0]._get_tree().export_graphviz(out_file=None, 
                                                                                         feature_names=X.columns,
                                                                                         class_names=['vhigh', 'high', 'med', 'low'],
                                                                                         filled=True,
                                                                                         rounded=True,
                                                                                         special_characters=True))
dot_data.view()

# Plot the decision tree structure for the first tree in the second forest
dot_data = graphviz.Source(forest_of_forests.estimators_[1][0]._get_tree().export_graphviz(out_file=None, 
                                                                                         feature_names=X.columns,
                                                                                         class_names=['vhigh', 'high', 'med', 'low'],
                                                                                         filled=True,
                                                                                         rounded=True,
                                                                                         special_characters=True))
dot_data.view()

# Plot the decision tree structure for the first tree in the third forest
dot_data = graphviz.Source(forest_of_forests.estimators_[2][0]._get_tree().export_graphviz(out_file=None, 
                                                                                         feature_names=X.columns,
                                                                                         class_names=['vhigh', 'high', 'med', 'low'],
                                                                                         filled=True,
                                                                                         rounded=True,
                                                                                         special_characters=True))
dot_data.view()

# Plot the forest of forests structure
fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(20, 15))
for i, forest in enumerate(forest_of_forests.estimators_):
    for j,tree in enumerate(forest):
        dot_data = graphviz.Source(tree._get_tree().export_graphviz(out_file=None, 
                                                                    feature_names=X.columns,
                                                                     class_names=['vhigh', 'high', 'med', 'low'],
                                                                     filled=True,
                                                                     rounded=True,
                                                                     special_characters=True))
        ax = axes[i][j]
        ax.axis('off')
        ax.text(0.5, 0.5, f'Tree {j + 1} in Forest {i + 1}', 
                ha='center', va='center', transform=ax.transAxes)
        dot_data.render(filename=f'tree_{i}_{j}', format='png', directory='./trees')
        img = plt.imread(f'./trees/tree_{i}_{j}.png')
        ax.imshow(img)
plt.show()