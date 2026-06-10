# 📧 SMS Spam Detection System

A Machine Learning project that automatically classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and the Naive Bayes algorithm.

The project leverages **TF-IDF Vectorization** and **Multinomial Naive Bayes** to build an efficient text classification model capable of detecting unwanted spam messages with high accuracy.

---

# 🚀 Project Overview

Spam messages are a major issue in digital communication. Businesses, telecom providers, and messaging platforms use machine learning models to automatically identify and filter spam content.

This project demonstrates a complete NLP workflow:

* Data Collection
* Text Processing
* Feature Extraction
* Model Training
* Model Evaluation
* Model Persistence

The system learns patterns from thousands of SMS messages and predicts whether a new message is spam or legitimate.

---

# 🎯 Features

✅ SMS Spam Detection

✅ TF-IDF Feature Engineering

✅ Multinomial Naive Bayes Classification

✅ Train/Test Dataset Split

✅ Classification Report Generation

✅ Model & Vectorizer Export

✅ Reusable Machine Learning Pipeline

---

# 🛠️ Technologies Used

| Technology              | Purpose                  |
| ----------------------- | ------------------------ |
| Python                  | Programming Language     |
| Pandas                  | Data Processing          |
| Scikit-Learn            | Machine Learning         |
| TF-IDF Vectorizer       | Text Feature Extraction  |
| Multinomial Naive Bayes | Classification Algorithm |
| Joblib                  | Model Serialization      |

---

# 📊 Dataset

This project uses the famous **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

Dataset contains:

* Spam Messages
* Legitimate Messages (Ham)

Each record consists of:

| Column | Description |
| ------ | ----------- |
| label  | ham / spam  |
| text   | SMS content |

Example:

```csv
label,text
ham,I'm going to the meeting now.
spam,Congratulations! You won a free iPhone.
```

---

# 🧠 Machine Learning Pipeline

```text
SMS Messages
      │
      ▼
Text Cleaning
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Feature Matrix
      │
      ▼
Multinomial Naive Bayes
      │
      ▼
Prediction
      │
      ▼
Spam / Ham
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/sms-spam-classifier.git

cd sms-spam-classifier
```

## Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate.bat
```

Linux / Mac:

```bash
source venv/bin/activate.bat
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python spam_classifier.py
```

The script will:

1. Download the SMS Spam Collection Dataset
2. Load and preprocess data
3. Split data into training and testing sets
4. Convert text into TF-IDF features
5. Train the Naive Bayes model
6. Evaluate model performance
7. Save trained artifacts

---

# 🔍 Feature Engineering

The model converts SMS messages into numerical vectors using:

### TF-IDF (Term Frequency – Inverse Document Frequency)

TF-IDF identifies important words while reducing the impact of frequently occurring but less informative words.

Example:

```text
"Congratulations! You won a free prize"
```

Important keywords:

* congratulations
* won
* free
* prize

These words receive higher weights during classification.

---

# 🎯 Example Predictions

| Message                           | Prediction |
| --------------------------------- | ---------- |
| Hey, are you free tonight?        | Ham        |
| Win a free vacation now!          | Spam       |
| Meeting starts at 10 AM           | Ham        |
| Claim your cash prize immediately | Spam       |

---

# 🌍 Real-World Applications

Spam detection systems are used by:

* Gmail
* Outlook
* Yahoo Mail
* WhatsApp
* Telegram
* SMS Gateways
* Telecom Companies

---

# 📚 Learning Outcomes

This project demonstrates:

* Natural Language Processing
* Text Classification
* Feature Engineering
* Machine Learning Workflow
* Model Evaluation
* Model Deployment Preparation

---

# 👨‍💻 Author

Himanshu Singh Yadav

Machine Learning Enthusiast | Python Developer | Data Science Learner

⭐ If you found this project useful, consider giving it a star on GitHub.
