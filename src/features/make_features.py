import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

numeric = ["area_in_sqft","beds","baths"]
categorical = ["type","furnishing","location"]

def build_features(df: pd.DataFrame, encoding: str = "onehot"):
    y = df["rent"]
    x = df[numeric + categorical].copy()
    if encoding == "onehot":
        x = pd.get_dummies(x, columns=categorical, drop_first=True)
    elif encoding == "label":
        for col in categorical:
            le = LabelEncoder()
            x[col] = le.fit_transform(x[col].astype(str))
    else:
        raise ValueError("Unsupported encoding type. Use 'onehot' or 'label'.")
    return x,y
    
def train_val_split(x,y, test_size = 0.2, random_state = 42):
    return train_test_split(x,y,test_size=test_size,random_state=random_state)