import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from preprocess import load_data, preprocess

df = load_data("data/train.csv")
df["clean"] = df["text"].apply(preprocess)
df["words"] = df["text"].str.split().str.len()

# 1. Class distribution
sns.countplot(data=df, x="label")
plt.title("Class Distribution")
plt.savefig("outputs/class_distribution.png"); plt.close()

# 2. Article length
sns.boxplot(data=df, x="label", y="words")
plt.title("Article Length by Topic")
plt.savefig("outputs/length.png"); plt.close()

# 3. Top words aur word cloud
for label, group in df.groupby("label"):
    text = " ".join(group["clean"])
    words, counts = zip(*Counter(text.split()).most_common(15))
    plt.barh(words[::-1], counts[::-1])
    plt.title(f"Top Words: {label}"); plt.tight_layout()
    name = label.replace("/", "_")
    plt.savefig(f"outputs/top_{name}.png"); plt.close()
    WordCloud(width=800, height=400, collocations=False).generate(text)\
        .to_file(f"outputs/cloud_{name}.png")