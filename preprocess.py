import re, html
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)

LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
STOP = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def basic_clean(text):
    text = re.sub(r"&?#39;", "'", str(text))
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)       # HTML tags
    text = text.replace("\\", " ")
    return re.sub(r"\s+", " ", text).strip()

def preprocess(text):
    text = basic_clean(text).lower()           # normalization
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)               # tokenization
    return " ".join(lemmatizer.lemmatize(t) for t in tokens
                    if t not in STOP and len(t) > 1)

def load_data(path):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Missing:\n", df.isnull().sum())

    df = df.dropna(subset=["Class Index"])
    df[["Title", "Description"]] = df[["Title", "Description"]].fillna("")
    df["text"] = (df["Title"] + " " + df["Description"]).apply(basic_clean)
    df["label"] = df["Class Index"].astype(int).map(LABELS)
    df = df.dropna(subset=["label"])           # invalid labels
    df = df[df["text"] != ""]

    # same text with different labels = inconsistent
    df = df[df.groupby("text")["label"].transform("nunique") == 1]
    print("Duplicates:", df.duplicated("text").sum())
    return df.drop_duplicates("text").reset_index(drop=True)