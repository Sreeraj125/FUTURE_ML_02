import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

raw_data = {
    "ticket_text": [
        "My account is locked and I cannot log into the dashboard. Help!",
        "Can I get a refund for my subscription? I was double charged this month.",
        "The mobile app keeps crashing every time I try to upload a profile photo.",
        "How do I update my billing address and change my credit card on file?",
        "Is there a dark mode available for the desktop version of the platform?",
        "Urgent: The entire production server is down and throwing 500 errors!",
        "I forgot my password and the reset link email is not arriving.",
        "Your pricing page is confusing. What is included in the Enterprise plan?",
    ],
    "category": [
        "Account",
        "Billing",
        "Technical Issue",
        "Billing",
        "General Query",
        "Technical Issue",
        "Account",
        "General Query",
    ],
    "priority": ["Medium", "High", "Medium", "Low", "Low", "High", "Medium", "Low"],
}

df = pd.DataFrame(raw_data)


def clean_support_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = text.split()
    stop_words = set(ENGLISH_STOP_WORDS)
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)


df["cleaned_text"] = df["ticket_text"].apply(clean_support_text)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_features = tfidf_vectorizer.fit_transform(df["cleaned_text"])

y_category = df["category"]
y_priority = df["priority"]

X_train, X_test, y_cat_train, y_cat_test, y_prio_train, y_prio_test = (
    train_test_split(
        X_features, y_category, y_priority, test_size=0.2, random_state=42
    )
)

category_model = LinearSVC(random_state=42)
category_model.fit(X_train, y_cat_train)

priority_model = LinearSVC(random_state=42)
priority_model.fit(X_train, y_prio_train)

y_cat_pred = category_model.predict(X_test)
y_prio_pred = priority_model.predict(X_test)

print("## MODEL EVALUATION REPORT ##\n" + "=" * 30)
print("\n--- Category Classification Performance ---")
print(classification_report(y_cat_test, y_cat_pred, zero_division=0))
print("\n--- Priority Classification Performance ---")
print(classification_report(y_prio_test, y_prio_pred, zero_division=0))


def classify_new_ticket(ticket_string):
    cleaned = clean_support_text(ticket_string)
    vectorized = tfidf_vectorizer.transform([cleaned])
    predicted_category = category_model.predict(vectorized)[0]
    predicted_priority = priority_model.predict(vectorized)[0]
    return {
        "Ticket Text": ticket_string,
        "Predicted Category": predicted_category,
        "Predicted Priority": predicted_priority,
    }


new_ticket = (
    "Hey! The server is completely down. Customers are seeing errors on checkout."
)
result = classify_new_ticket(new_ticket)

print("\n" + "=" * 50)
print("LIVE TICKET INFERENCE DEMO")
print("=" * 50)
for key, value in result.items():
    print(f"{key}: {value}")