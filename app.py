import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# -------------------------------
# UI
# -------------------------------

st.title("🔢 NLP Feature Engineering App")

st.write("Convert text into numerical features using BoW, TF-IDF, and N-grams.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Dataset Preview")
    st.dataframe(df.head())

    # Select text column
    text_column = st.selectbox("Select Text Column", df.columns)

    # -------------------------------
    # Feature Engineering Options
    # -------------------------------

    st.write("## ⚙️ Feature Settings")

    feature_method = st.selectbox(
        "Choose Representation Method",
        ["Bag of Words", "TF-IDF"]
    )

    ngram_option = st.selectbox(
        "Select N-gram Range",
        ["Unigram (1,1)", "Bigram (2,2)", "Trigram (3,3)", "Unigram + Bigram (1,2)"]
    )

    max_features = st.slider("Max Features", 10, 5000, 100)

    # Map n-grams
    ngram_map = {
        "Unigram (1,1)": (1, 1),
        "Bigram (2,2)": (2, 2),
        "Trigram (3,3)": (3, 3),
        "Unigram + Bigram (1,2)": (1, 2)
    }

    selected_ngram = ngram_map[ngram_option]

    # -------------------------------
    # Apply Feature Engineering
    # -------------------------------

    if st.button("🚀 Generate Features"):

        corpus = df[text_column].astype(str).tolist()

        # Vectorizer selection
        if feature_method == "Bag of Words":
            vectorizer = CountVectorizer(
                ngram_range=selected_ngram,
                max_features=max_features
            )
        else:
            vectorizer = TfidfVectorizer(
                ngram_range=selected_ngram,
                max_features=max_features
            )

        X = vectorizer.fit_transform(corpus)

        feature_names = vectorizer.get_feature_names_out()

        feature_df = pd.DataFrame(X.toarray(), columns=feature_names)

        # -------------------------------
        # Output
        # -------------------------------

        st.write("### 📊 Feature Matrix")
        st.dataframe(feature_df.head())

        st.write("### 🔝 Top Features")

        top_features = feature_df.sum().sort_values(ascending=False).head(20)
        st.dataframe(top_features)

        # Download option
        csv = feature_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Feature Matrix",
            data=csv,
            file_name="features.csv",
            mime="text/csv"
        )