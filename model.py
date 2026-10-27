import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Load the dataset
df = pd.read_csv('StressLevelDataset.csv')

# ===============================
# Exploratory Data Analysis (EDA)
# ===============================

# 1. Distribution of Stress Levels
plt.figure(figsize=(6, 4))
sns.countplot(x='stress_level', data=df, palette='Set2')
plt.title("Distribution of Stress Levels")
plt.xlabel("Stress Level")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 3. Pairplot (if not too many features)
# Uncomment if the dataset is small
# sns.pairplot(df, hue='stress_level', diag_kind='kde', palette='husl')
# plt.suptitle("Pairwise Feature Relationships by Stress Level", y=1.02)
# plt.show()

# 4. Boxplots for numeric features
numeric_cols = df.select_dtypes(include='number').columns.drop('stress_level')
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='stress_level', y=col, data=df, palette='Set3')
    plt.title(f"{col} Distribution by Stress Level")
    plt.tight_layout()
    plt.show()

# 5. Violin plots
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.violinplot(x='stress_level', y=col, data=df, palette='Pastel1')
    plt.title(f"{col} Violin Plot by Stress Level")
    plt.tight_layout()
    plt.show()

# 6. Bar plots for categorical features
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue='stress_level', data=df, palette='Set1')
    plt.title(f"Stress Level Distribution by {col}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# =====================
# Model Training & Eval
# =====================

# Prepare features and target
X = df.drop(columns='stress_level')
y = df['stress_level']

# Encode categorical features if present
X = pd.get_dummies(X, drop_first=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for Decision Tree")
plt.savefig("confusion_matrix.png")
plt.show()

# Classification report as DataFrame
report_dict = classification_report(y_test, y_pred, output_dict=True)
# Display only known classes (assuming 0, 1, 2 map to 'Low', 'Moderate', 'High')
metrics_df = pd.DataFrame(report_dict).transpose().loc[['0', '1', '2'], ['precision', 'recall', 'f1-score']]

# Plot metrics
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title("Classification Metrics by Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
