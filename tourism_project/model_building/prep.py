import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(input_path="tourism_project/data/tourism.csv"):
    df = pd.read_csv(input_path)

    # Drop identifier / leftover index columns — not predictive
    df = df.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore")

    # Drop rows with a missing target (can't train on those)
    df = df.dropna(subset=["ProdTaken"])

    # --- Fix dirty categorical values found in the actual dataset ---
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Defensive cleaning (this file has zero nulls, but a resubmitted CSV may not)
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Saved at repo root to match the artifact paths in pipeline.yml
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"Train shape: {Xtrain.shape}, Test shape: {Xtest.shape}")
    print("✅ Train/test splits saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
    return Xtrain, Xtest, ytrain, ytest

if __name__ == "__main__":
    prepare_data()
