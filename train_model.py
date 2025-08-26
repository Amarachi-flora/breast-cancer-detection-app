# ============================================================
#  ZION TECH HUB - Breast Cancer Detection Project
#  File: train_model.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Step 1: Load and Prepare Data ---
df = pd.read_csv("breast_cancer.csv")
df.drop(columns=['id'], errors='ignore', inplace=True)
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Step 2: EDA - Boxplot of Selected Features ---
features_to_plot = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[features_to_plot], palette="pastel")
plt.title("Boxplot of Tumor Features")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Step 3: EDA - Diagnosis Class Distribution ---
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="diagnosis", data=df, palette="Set2")
plt.xticks([0, 1], ["Benign", "Malignant"])
plt.title("Distribution of Diagnosis Classes")
plt.xlabel("Diagnosis")
plt.ylabel("Count")

# Add data labels
for p in ax.patches:
    ax.annotate(f"{p.get_height()}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 7),
                textcoords='offset points')
plt.tight_layout()
plt.show()

# --- Step 4: EDA - Correlation Heatmap ---
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.show()

# --- Step 5: Model Training and Comparison ---
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = []
for name, model in models.items() :
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)

# --- Step 6: Bar Charts of Model Performance ---
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
colors = ["#2a9d8f", "#264653", "#e76f51", "#f4a261"]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Performance Comparison of ML Models", fontsize=18)

for i, metric in enumerate(metrics):
    row, col = i // 2, i % 2
    ax = axs[row][col]
    bars = ax.bar(results_df["Model"], results_df[metric], color=colors[i])

    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}",
                ha='center', fontweight='bold')
    ax.set_title(f"{metric} by Model", fontsize=14)
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df["Model"], rotation=45)

plt.tight_layout()
plt.show()

# --- Step 7: Confusion Matrix for Best Model ---
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix: {best_model_name}")
plt.grid(False)
plt.tight_layout()
plt.show()

# --- Step 8: Hyperparameter Tuning - Random Forest ---
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# --- Step 9: Visualize Grid Search Results ---
cv_results = pd.DataFrame(grid_search .cv_results_)
pivot_table = cv_results.pivot_table(
    index='param_n_estimators',
    columns='param_max_depth',
    values='mean_test_score'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("GridSearchCV F1-Score Heatmap\n(n_estimators vs max_depth)")
plt.xlabel("Max Depth")
plt.ylabel("N Estimators")
plt.tight_layout()
plt.show()

# --- Step 10: Save Final Best Random Forest (Top 10 Features) ---
best_rf_model = grid_search.best_estimator_
importances = pd.Series(best_rf_model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

top_features = importances_sorted.head(10).index.tolist()
print("\n Top 10 Features Selected:")
for feature, score in importances_sorted.head(10).items():
    print(f"   - {feature}: {score:.4f}")

#  Retrain using ONLY top 10 features
best_rf_model.fit(X[top_features], y)

# Save model + features
os.makedirs("breast_cancer_ml_project/models", exist_ok=True)
joblib.dump(best_rf_model, "breast_cancer_ml_project/models/best_rf_model.pkl")
joblib.dump(top_features, "breast_cancer_ml_project/models/top_features.pkl")

print("\n Final Random Forest (Top 10 features only) saved successfully")

# --- Step 11: Plot Top Feature Importances ---
plt.figure(figsize=(8, 5))
sns.barplot(x=importances_sorted.head(10).values, y=importances_sorted.head(10).index, palette="viridis")
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Project Workflow Steps ---
steps = [
    "Data Preparation",
    "Models Selection",
    "Training the Models",
    "Fine-Tuning the Model",
    "Evaluating the Models",
    "AUC-ROC Analysis",
    "Deployment (Streamlit App)"
]

# --- Create Figure ---
fig, ax = plt.subplots(figsize=(14, 2))  # made figure slightly wider

# --- Draw Rectangles and Arrows ---
box_width = 2.3   # increased from 1.9 to 2.3
for i, step in enumerate(steps):
    ax.add_patch(
        mpatches.Rectangle(
            (i*2.4, 0), box_width, 0.8, facecolor="#f48fb1", edgecolor="black", linewidth=1.2
        )
    )
    ax.text(i*2.4 + box_width/2, 0.4, step, ha="center", va="center", 
            fontsize=9, weight="bold", wrap=True)
    
    # Draw arrows between boxes
    if i < len(steps)-1:
        ax.annotate(
            "", xy=(i*2.4+box_width, 0.4), xytext=(i*2.4+box_width+0.1, 0.4),
            arrowprops=dict(arrowstyle="->", lw=2, color="black")
        )

# --- Cleanup ---
ax.set_xlim(-0.5, len(steps)*2.6)  # adjusted x-limit to fit longer boxes
ax.set_ylim(-0.5, 1.5)
ax.axis("off")
plt.title("Project Workflow: Breast Cancer Detection", fontsize=14, weight="bold")

# --- Save & Show ---
plt.tight_layout()
plt.savefig("workflow.png", dpi=300)
plt.show()

