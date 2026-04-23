import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset (you can expand this)
data = {
    "email": [
        "Congratulations! You won a lottery. Click here now",
        "Urgent! Your account is hacked. Reset password immediately",
        "Meeting scheduled at 10 AM tomorrow",
        "Project submission deadline extended",
        "Win cash prizes now!!! Click link",
        "Your invoice is attached",
        "Verify your account now or it will be suspended",
        "Lunch at 1 PM?"
    ],
    "label": [1, 1, 0, 0, 1, 0, 1, 0]  # 1 = phishing, 0 = safe
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["email"], df["label"], test_size=0.3, random_state=42
)

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Test with user input
while True:
    msg = input("\nEnter email text (or type 'exit'): ")
    if msg.lower() == "exit":
        break

    msg_vec = vectorizer.transform([msg])
    result = model.predict(msg_vec)[0]

    if result == 1:
        print("⚠️ Phishing Email Detected")
    else:
        print("✅ Safe Email")
