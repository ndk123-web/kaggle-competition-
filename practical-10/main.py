import pandas as pd
import numpy as np
import argparse

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# -----------------------------
# STEP 1: Load dataset properly
# -----------------------------
def load_data(file_path):
    """
    Load wine.data file and assign column names
    """
    columns = [
        "class",
        "Alcohol", "Malic acid", "Ash", "Alcalinity of ash",
        "Magnesium", "Total phenols", "Flavanoids",
        "Nonflavanoid phenols", "Proanthocyanins",
        "Color intensity", "Hue",
        "OD280/OD315 of diluted wines", "Proline"
    ]

    df = pd.read_csv(file_path, header=None, names=columns)
    return df


# -----------------------------
# STEP 2: Preprocessing
# -----------------------------
def preprocess(df, target_col):
    """
    Split features and target + standardize features
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# -----------------------------
# STEP 3: Apply LDA
# -----------------------------
def apply_lda(X, y, n_components):
    """
    Apply LDA with given components
    """
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    return X_lda


# -----------------------------
# STEP 4: Classification
# -----------------------------
def evaluate_model(X, y):
    """
    Train Logistic Regression and return accuracy + F1 score
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return acc, f1


# -----------------------------
# STEP 5: Save LDA projection
# -----------------------------
def save_projection(X_lda, y, filename):
    """
    Save LD1, LD2 projection to CSV
    """
    df = pd.DataFrame()

    # Handle 1D or 2D case
    df["LD1"] = X_lda[:, 0]

    if X_lda.shape[1] > 1:
        df["LD2"] = X_lda[:, 1]

    df["Class"] = y.values
    df["SampleId"] = np.arange(len(y))

    df.to_csv(filename, index=False)
    print(f"[INFO] Saved projection to {filename}")


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--target", default="class", help="Target column name")

    args = parser.parse_args()

    # Load data
    df = load_data(args.data)

    # Preprocess
    X, y = preprocess(df, args.target)

    # -----------------------------
    # ORIGINAL DATA PERFORMANCE
    # -----------------------------
    orig_acc, orig_f1 = evaluate_model(X, y)

    print("\n=== ORIGINAL DATA ===")
    print(f"Accuracy: {orig_acc:.4f}")
    print(f"F1 Score: {orig_f1:.4f}")

    # -----------------------------
    # LDA (1 COMPONENT)
    # -----------------------------
    X_lda1 = apply_lda(X, y, n_components=1)
    lda1_acc, lda1_f1 = evaluate_model(X_lda1, y)

    print("\n=== LDA (1 COMPONENT) ===")
    print(f"Accuracy: {lda1_acc:.4f}")
    print(f"F1 Score: {lda1_f1:.4f}")

    # -----------------------------
    # LDA (2 COMPONENTS)
    # -----------------------------
    X_lda2 = apply_lda(X, y, n_components=2)
    lda2_acc, lda2_f1 = evaluate_model(X_lda2, y)

    print("\n=== LDA (2 COMPONENTS) ===")
    print(f"Accuracy: {lda2_acc:.4f}")
    print(f"F1 Score: {lda2_f1:.4f}")

    # Save projection (2D)
    save_projection(X_lda2, y, "lda_projection.csv")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()