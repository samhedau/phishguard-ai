import streamlit as st
import joblib

# ------------------ LOAD MODEL ------------------
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ------------------ HEADER ------------------
st.title("📧 AI Email Spam / Phishing Detector")
st.caption("Detect whether an email is SAFE or SPAM using Machine Learning")

st.divider()

# ------------------ INPUT ------------------
text = st.text_area("✉️ Paste your email content here", height=200)

# ------------------ PREDICT BUTTON ------------------
if st.button("🔍 Analyze Email"):

    if text.strip() == "":
        st.warning("Please enter email text")
    else:
        vec = vectorizer.transform([text])

        pred = model.predict(vec)[0]
        score = model.decision_function(vec)[0]

        confidence = abs(score)

        st.divider()

        # ------------------ RESULT ------------------
        if pred == 1:
            st.error("🚨 SPAM / PHISHING EMAIL DETECTED")
        else:
            st.success("✅ SAFE EMAIL")

        # ------------------ SCORE DISPLAY ------------------
        st.subheader("📊 Analysis Details")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Model Score", value=f"{score:.2f}")

        with col2:
            st.metric(label="Confidence (raw)", value=f"{confidence:.2f}")

        # ------------------ EXPLANATION ------------------
        st.info(
            "💡 Note: Score > 0 means SPAM, Score < 0 means SAFE. "
            "This model is based on TF-IDF + Linear SVM."
        )

# ------------------ FOOTER ------------------
st.divider()
st.caption("Built using Machine Learning + Streamlit 🚀")