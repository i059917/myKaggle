"""Titanic - Logistic Regression baseline.

Kaggle competition: https://www.kaggle.com/competitions/titanic
Goal: predict Survived (0/1) from passenger features.
"""

from typing import Any


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into numeric features.

    Every feature below is a number so Logistic Regression can use it:
    logistic regression computes z = b0 + b1*x1 + ... and cannot read text.

    Note: Fare is left as NaN on purpose so SimpleImputer inside the pipeline
    learns its fill value from training folds only, instead of from all rows.
    """
    X = pd.DataFrame(index=df.index)

    # Pclass is already 1/2/3 -> keep as is (ordinal).
    X["Pclass"] = df["Pclass"]

    # Sex: female=1, male=0.
    X["Sex"] = (df["Sex"] == "female").astype(int)

    # Age: fill by the median of each (Pclass, Sex) group, because a third-class
    # passenger and a first-class passenger are not the same age. transform()
    # broadcasts the 6 group medians back to all rows, so the result aligns 1:1
    # with df. The second fillna is a fallback for a group with no ages at all.
    X["Age"] = df["Age"]
    group_median = X.groupby(["Pclass", "Sex"])["Age"].transform("median")
    X["Age"] = X["Age"].fillna(group_median).fillna(X["Age"].median())

    # Fare: 1 missing value in test.csv, filled by the pipeline.
    X["Fare"] = df["Fare"]

    # Family: siblings/spouses + parents/children, plus a "traveling alone" flag.
    X["SibSp"] = df["SibSp"]
    X["Parch"] = df["Parch"]
    X["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

    # Embarked: one-hot (drop the first category to avoid redundancy).
    X = pd.concat([X, pd.get_dummies(df["Embarked"], prefix="Embarked", dtype=int).iloc[:, 1:]], axis=1)

    return X


def make_model() -> Pipeline:
    """Build the pipeline: impute -> scale -> classify.

    Every step that learns from data (impute, scale) lives inside the pipeline,
    so cross-validation refits them on each training fold instead of once on the
    full dataset. That keeps validation folds unseen during preprocessing.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def main() -> None:
    train, test = load_data()

    X = build_features(train)
    y = train["Survived"]

    model = make_model()

    # 5-fold cross-validation: the pipeline is split as a whole, so imputation
    # and scaling are refitted per fold on training data only.
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    for i, score in enumerate[Any](scores, start=1):
        print("  Fold %d: %.4f" % (i, score))
    print("Cross-validation accuracy: %.4f (+/- %.4f)" % (scores.mean(), scores.std()))

    # Refit on the full training set for the submission. The pipeline learns
    # medians and means from all 891 rows, which is correct here: test rows are
    # still unseen, and this is the model we actually ship.
    model.fit(X, y)

    # Show the math: z = intercept + sum(coefficient_i * feature_i), p = 1 / (1 + e^-z)
    clf = model.named_steps["clf"]
    print("\nintercept (b0): %.4f" % clf.intercept_[0])
    print("coefficients (b1..bn), sorted by impact:")
    coefs = pd.Series(clf.coef_[0], index=X.columns).sort_values(key=np.abs, ascending=False)
    for name, value in coefs.items():
        print("  %-16s % .4f" % (name, value))

    # Predict on the unseen test set. transform() reuses the values learned
    # from the 891 training rows; it never looks at test statistics.
    predictions = model.predict(build_features(test))

    OUT_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    print("\nWrote %d rows to %s" % (len(submission), OUT_DIR / "submission.csv"))


if __name__ == "__main__":
    main()