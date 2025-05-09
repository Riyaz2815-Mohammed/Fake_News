import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

#plt.style.use('seaborn-darkgrid')
plt.style.use('ggplot')  # or 'default'

# ---------------------- TITLE ----------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection Web App")

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("About")
st.sidebar.info("🚀 This app detects whether a news article is **Fake** or **Real** using Natural Language Processing and Machine Learning.\n\nBuilt with ❤️ using Streamlit & Scikit-learn.")

# ---------------------- FILE UPLOAD ----------------------
true_file = st.file_uploader("Upload True News CSV (Real)", type="csv")
false_file = st.file_uploader("Upload Fake News CSV (Fake)", type="csv")

# ---------------------- MAIN PROCESS ----------------------
if true_file and false_file:
    true_df = pd.read_csv(true_file)
    false_df = pd.read_csv(false_file)

    # Add labels
    true_df['label'] = 1
    false_df['label'] = 0

    # Combine and shuffle
    df = pd.concat([true_df, false_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    st.subheader("📊 Sample Data")
    st.write(df[['title', 'text', 'label']].head())

    # Features and Labels
    X = df['text']
    y = df['label']

    # Vectorization
    vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
    X_tfidf = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Model Training
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ---------------------- EVALUATION ----------------------
    st.subheader("📈 Model Evaluation")
    st.text(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    st.text(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("📉 Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'], ax=ax)
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    # ---------------------- USER INPUT ----------------------
    st.subheader("📝 Test a News Article")
    user_input = st.text_area("Enter the news content below:")
    if st.button("Check if Real or Fake"):
        if user_input.strip():
            input_tfidf = vectorizer.transform([user_input])
            prediction = model.predict(input_tfidf)[0]
            if prediction == 1:
                st.success("✅ This appears to be **Real News**.")
            else:
                st.error("❌ This appears to be **Fake News**.")
        else:
            st.warning("Please enter some text to check.")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.markdown("Made with ❤️ by | Powered by `Streamlit` & `Scikit-learn`")
