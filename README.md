# Haryana News - News Topic Classification

An NLP-based News Topic Classification web application that automatically classifies news headlines and articles into four categories: **World, Sports, Business, and Sci/Tech**.

Built using Python, Natural Language Processing (NLP), Machine Learning, and Streamlit, this project allows users to classify individual news articles as well as multiple articles through CSV file uploads.

**GitHub Repository:** [NLP-Project](https://github.com/ikhushichauhan/NLP-Project)  
**Live Demo:** [Haryana News](https://haryana-news.streamlit.app/)

---

## Project Overview

With the increasing volume of digital news, manually organizing articles into relevant categories can be time-consuming.

Haryana News addresses this problem using Natural Language Processing and Machine Learning techniques. It analyzes the textual content of a news article and predicts its most relevant category.

The application provides an interactive interface where users can:
- Enter a news headline or article for classification.
- Get an automatically predicted news category.
- View prediction confidence.
- Upload CSV files to classify multiple news articles.
- Download the results of batch classification.

## Objectives

- Build an automated news classification system using NLP.
- Perform text cleaning and preprocessing to improve data quality.
- Extract meaningful textual features using different vectorization techniques.
- Compare multiple Machine Learning algorithms.
- Identify the best-performing model using evaluation metrics.
- Develop an interactive web application using Streamlit.

## Features

### 1. Home Page
- Project introduction and overview.
- Sample news stories.
- Easy navigation between application sections.

### 2. Single News Classification
- Enter a headline or complete news article.
- Get an instant predicted category.
- View prediction confidence.

### 3. Batch CSV Classification
- Upload a CSV file containing a `text` column.
- Classify multiple news articles at once.
- Download the CSV containing predicted categories.

### 4. About Page
- Project information.
- Technology stack.
- Model and methodology overview.

## News Categories

The model classifies news articles into four categories:

| Category | Description |
|---|---|
| World | International news and global events |
| Sports | Sports events, teams, and players |
| Business | Business, finance, and economic news |
| Sci/Tech | Science, technology, and innovation |

## Dataset

This project uses the **AG News dataset**, containing news articles across four categories.

- Training dataset: 120,000 articles.
- Test dataset: 7,600 articles.
- Input text: Title and Description.
- Target: News Category.

The project repository includes:
- `data/train.csv`
- `data/test.csv`

## Methodology

The project follows a complete NLP and Machine Learning pipeline.

### 1. Data Preprocessing

The raw news text is cleaned and prepared before model training.

Preprocessing steps include:

- Handling missing and invalid values.
- Removing duplicate records and resolving conflicting labels.
- Combining Title and Description.
- Removing HTML content.
- Converting text to lowercase.
- Removing URLs and unwanted characters.
- Tokenization.
- Removing English stopwords.
- Lemmatization.

### 2. Feature Extraction

Different text representation techniques were explored:

- Bag of Words (BoW)
- TF-IDF (Term Frequency–Inverse Document Frequency)
- Word2Vec

### 3. Model Training

Multiple Machine Learning models were trained and compared:

- Multinomial Naive Bayes with Bag of Words.
- Multinomial Naive Bayes with TF-IDF.
- Logistic Regression with TF-IDF.
- Linear Support Vector Classifier (LinearSVC) with TF-IDF and probability calibration.
- Logistic Regression with Word2Vec.

### 4. Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

The best-performing model was selected based on validation performance and saved for application use.

## Model Comparison

Validation results from the model comparison:

| Model | Accuracy | Macro F1-Score |
|---|---:|---:|
| BoW + Naive Bayes | 90.42% | 90.40% |
| TF-IDF + Naive Bayes | 90.67% | 90.64% |
| TF-IDF + Logistic Regression | 91.77% | 91.76% |
| **TF-IDF + SVM** | **92.02%** | **92.01%** |
| Word2Vec + Logistic Regression | 88.23% | 88.21% |

**Best Model: TF-IDF + Calibrated LinearSVC**

- Validation Accuracy: **92.02%**
- Macro Precision: **92.02%**
- Macro Recall: **92.02%**
- Macro F1-Score: **92.01%**

The selected model is saved as `models/news_model.joblib` and loaded by the Streamlit application.

*Note: These are validation results from the model comparison, not a claim of final test-set performance.*

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data handling and numerical operations |
| NLTK | Text preprocessing |
| Scikit-learn | Feature extraction, model training, and evaluation |
| Gensim | Word2Vec embeddings |
| Joblib | Model serialization |
| Matplotlib & Seaborn | Data visualization |
| WordCloud | Text visualization |
| Streamlit | Interactive web application |
| Pytest | Testing |

## Project Structure

```text
NLP-Project/
│
├── app.py
├── preprocess.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── news_model.joblib
│
├── images/
│   ├── news.png
│   └── vardaat.png
│
└── outputs/
    ├── model_comparison.csv
    └── [EDA and evaluation visualizations]
```

## Getting Started

Follow these steps to run the project locally.

### Prerequisites

Make sure you have installed:

- Python 3.10 or a compatible version.
- Git
- Visual Studio Code (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/ikhushichauhan/NLP-Project.git
```

### 2. Navigate to the Project Directory

```bash
cd NLP-Project
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## CSV Input Format

For batch classification, upload a CSV file containing a column named `text`.

Example:

| text |
|---|
| India wins an international cricket match |
| Global markets react to economic changes |
| New technology improves communication |

The application generates predictions for the uploaded articles and allows you to download the results.

## Deployment

The application is deployed using **Streamlit Community Cloud**.

**Live Application:** https://haryana-news.streamlit.app/

## Future Enhancements

- Support for additional news categories.
- Multilingual news classification.
- Integration with live news APIs.
- Advanced deep learning models such as LSTM and BERT.
- Improved explainability to highlight important words influencing predictions.
- News summarization and sentiment analysis.
- Enhanced analytics dashboard for classification insights.

## Author

**Khushi Chauhan**

BCA Student | NLP & Machine Learning Enthusiast

- GitHub: [@ikhushichauhan](https://github.com/ikhushichauhan)

## License

This project is developed for educational and academic purposes.
