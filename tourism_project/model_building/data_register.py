import pandas as pd

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch"
]
# Note: the raw CSV also ships an "Unnamed: 0" index column that is not
# part of the data dictionary — it shows up as an "extra" column and is
# dropped later in prep.py, not treated as a validation failure here.

def register_dataset(path="tourism_project/data/tourism.csv"):
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)

    print("=== Data Registration Summary ===")
    print(f"File: {path}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Missing expected columns: {missing if missing else 'None'}")
    print(f"Unexpected extra columns: {extra if extra else 'None'}")
    print(f"Nulls per column:\n{df.isnull().sum()}")
    print(f"Target distribution (ProdTaken):\n{df['ProdTaken'].value_counts()}")

    if missing:
        raise ValueError(f"Dataset failed validation. Missing columns: {missing}")
    print("\n✅ Dataset validated successfully.")
    return df

if __name__ == "__main__":
    register_dataset()
