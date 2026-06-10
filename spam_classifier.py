
"""
spam_classifier.py

Downloads the SMS Spam Collection dataset (UCI), trains a Naive Bayes text classifier
using TF-IDF features, evaluates it, and saves a simple model and vectorizer.

Usage:
    python spam_classifier.py

Outputs:
 - trained_model.joblib
 - tfidf_vectorizer.joblib
 - evaluation_report.txt
 - sms_spam_collection.csv (downloaded dataset)
 
Note: This script will download the dataset from UCI. If you prefer to provide the file locally,
place 'SMSSpamCollection' or 'sms_spam_collection.csv' in the same folder.
"""

import os
import requests
import zipfile
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

def download_and_extract(dest_folder="."):
    print("Downloading dataset from UCI...")
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dest_folder)
    print("Extracted files:", z.namelist())

def load_dataset(path="."):
    # The UCI archive contains file 'SMSSpamCollection' (tab-separated: label \t text)
    local = os.path.join(path, "SMSSpamCollection")
    if not os.path.exists(local):
        # check csv variant
        csvp = os.path.join(path, "sms_spam_collection.csv")
        if os.path.exists(csvp):
            df = pd.read_csv(csvp)
            # expected columns: label,text or v1,v2
            if 'v1' in df.columns and 'v2' in df.columns:
                df = df.rename(columns={'v1':'label','v2':'text'})
            return df[['label','text']]
        else:
            raise FileNotFoundError("Dataset file not found. Please run download or place SMSSpamCollection file in folder.")
    # read tab separated file
    df = pd.read_csv(local, sep='\t', header=None, names=['label','text'], quoting=3)
    return df

def preprocess_and_train(df, output_folder="."):
    df['label_num'] = df.label.map({'ham':0, 'spam':1})
    X = df['text'].astype(str)
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vect = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=2)
    X_train_tfidf = vect.fit_transform(X_train)
    X_test_tfidf = vect.transform(X_test)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train_tfidf, y_train)

    preds = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=['ham','spam'])
    cm = confusion_matrix(y_test, preds)

    # save artifacts
    joblib.dump(clf, os.path.join(output_folder, "trained_model.joblib"))
    joblib.dump(vect, os.path.join(output_folder, "tfidf_vectorizer.joblib"))

    with open(os.path.join(output_folder, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write("Accuracy: {:.4f}\n\n".format(acc))
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    print("Training complete. Accuracy:", acc)
    print("Saved model, vectorizer, and evaluation report to", output_folder)

def main():
    try:
        download_and_extract(".")
    except Exception as e:
        print("Download failed or skipped:", e)
    df = load_dataset(".")
    print("Dataset loaded. Examples:", len(df))
    preprocess_and_train(df, output_folder=".")

if __name__ == "__main__":
    main()
