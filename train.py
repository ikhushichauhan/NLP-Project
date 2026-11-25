import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, ConfusionMatrixDisplay)
from preprocess import load_data, preprocess

train_df = load_data("data/train.csv")
test_df = load_data("data/test.csv")
test_df = test_df[~test_df["text"].isin(train_df["text"])].copy()  # leakage rokne ke liye
for d in (train_df, test_df):
    d["clean"] = d["text"].apply(preprocess)

X_tr, X_val, y_tr, y_val = train_test_split(
    train_df["clean"], train_df["label"], test_size=0.2,
    stratify=train_df["label"], random_state=42)

def scores(name, y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return {"Model": name, "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": p, "Recall": r, "F1": f1}

def tfidf():
    return TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)

models = {
    "BoW + NB": Pipeline([("vec", CountVectorizer(max_features=50000)),
                          ("clf", MultinomialNB())]),
    "TF-IDF + NB": Pipeline([("vec", tfidf()), ("clf", MultinomialNB())]),
    "TF-IDF + LogReg": Pipeline([("vec", tfidf()),
                                 ("clf", LogisticRegression(max_iter=1000))]),
    # Calibration se SVM bhi probability deta hai
    "TF-IDF + SVM": Pipeline([("vec", tfidf()),
                              ("clf", CalibratedClassifierCV(LinearSVC()))]),
}

results = []
for name, model in models.items():
    model.fit(X_tr, y_tr)
    results.append(scores(name, y_val, model.predict(X_val)))

# Word2Vec: har article = uske word vectors ka average
sentences = [t.split() for t in X_tr]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, seed=42)

def vectorize(texts):
    rows = []
    for t in texts:
        v = [w2v.wv[w] for w in t.split() if w in w2v.wv]
        rows.append(np.mean(v, axis=0) if v else np.zeros(100))
    return np.array(rows)

w2v_clf = LogisticRegression(max_iter=2000).fit(vectorize(X_tr), y_tr)
results.append(scores("Word2Vec + LogReg", y_val, w2v_clf.predict(vectorize(X_val))))

table = pd.DataFrame(results).sort_values("F1", ascending=False)
print(table)
table.to_csv("outputs/model_comparison.csv", index=False)

# GUI ke liye best Bag of Words / TF-IDF pipeline
best_name = table[table["Model"].isin(models.keys())].iloc[0]["Model"]
best = models[best_name]
best.fit(train_df["clean"], train_df["label"])

pred = best.predict(test_df["clean"])
print("\nSelected:", best_name)
print(classification_report(test_df["label"], pred, digits=4))
ConfusionMatrixDisplay.from_predictions(test_df["label"], pred)
plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")

joblib.dump(best, "models/news_model.joblib")
print("Model saved")