import streamlit as st
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
from nltk import FreqDist
import textstat
import spacy
spacy.cli.download("en_core_web_sm")
from spacy import displacy

# Set up Streamlit app
st.set_page_config(page_title="TextTrac: Navigate Text Data with AutoNLP", page_icon="📚", layout="wide")

st.title("TextTrac: Navigate Text Data with AutoNLP")

page = st.sidebar.radio("**Select a Page**", ["Home Page 🏠", "Tokenization 🔠", "Stopwords Removal 🛑", "Stemming 🌱", "Lemmatization 🌿", "POS Tagging 🏷️", "Dependency Parsing 🧩", "Word Cloud ☁️", "N-Grams 🔢", "Keyword Extraction 🔑", "Synonym and Antonym Detection 🔍", "Text Similarity 🔄", "Text Complexity Analysis 📊"])

# Function to load the CSS file
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
load_css('styles.css')

# Function to tokenize text
@st.cache_resource
def tokenize_text(text, tokenization_type):
    if tokenization_type == "Word Tokenization":
        tokens = word_tokenize(text)
    else:
        tokens = sent_tokenize(text)
    return tokens

# Function to remove stopwords
@st.cache_resource
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text if word.lower() not in stop_words]
    return filtered_text

# Function to perform stemming
@st.cache_resource
def perform_stemming(text):
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(word) for word in text]
    return stemmed_text

# Function to perform lemmatization
@st.cache_resource
def perform_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text]
    return lemmatized_text

# Function for Part-of-Speech (POS) tagging
@st.cache_resource
def pos_tagging(text):
    pos_tags = nltk.pos_tag(text)
    return pos_tags

# Function to create a word cloud
@st.cache_resource
def generate_word_cloud(text):
    if len(text) == 0:
        st.warning("Cannot generate a word cloud from empty text.")
    else:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(text))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

# Function to create n-grams
@st.cache_resource
def create_ngrams(tokens, n):
    n_grams = list(ngrams(tokens, n))
    return n_grams

# Function to generate n-grams text
@st.cache_resource
def generate_ngrams_text(n_grams):
    n_grams_text = [" ".join(gram) for gram in n_grams]
    return n_grams_text

# Function to extract keywords
@st.cache_resource
def extract_keywords(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    
    # Calculate word frequency
    word_freq = FreqDist(filtered_words)

    # Create a DataFrame with keywords and frequencies
    keywords_df = pd.DataFrame(word_freq.items(), columns=['Keyword', 'Frequency'])
    keywords_df = keywords_df.sort_values(by='Frequency', ascending=False)
    
    # Display keywords and their frequencies
    st.subheader("Keywords and Their Frequencies (Dataframe):")
    st.dataframe(keywords_df)

    csv = keywords_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
    href = f'data:file/csv;base64,{b64}'
    st.markdown(f'<a href="{href}" download="keyword_extraction_content.csv">Click here to download the document with Keywords and Their Frequencies</a>', unsafe_allow_html=True)
    
    # Plot keyword frequency distribution
    st.subheader("Keywords and Their Frequencies (Visualization Plot):")
    plt.figure(figsize=(10, 5))
    word_freq.plot(20, cumulative=False)
    st.pyplot(plt)

# Function to calculate text similarity
@st.cache_resource
def calculate_similarity(text1, text2):
    # Tokenize the input texts
    tokens = word_tokenize(text1 + " " + text2)
    
    # Create TF-IDF vectors for the texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1]).flatten()[0]
    return similarity_score

# Function to find synonyms and antonyms
def find_synonyms_antonyms(word):
    synonyms = set()
    antonyms = set()
    
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())
    
    return list(synonyms), list(antonyms)

# Function to process the text and find synonyms and antonyms for each word
def process_text_for_synonyms_antonyms(text):
    tokens = word_tokenize(text)
    results = []
    for token in tokens:
        synonyms, antonyms = find_synonyms_antonyms(token)
        results.append((token, synonyms, antonyms))
    return results

# Function to perform text complexity analysis
def analyze_text_complexity(text):
    return {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Smog Index": textstat.smog_index(text),
        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(text),
        "Coleman-Liau Index": textstat.coleman_liau_index(text),
        "Automated Readability Index": textstat.automated_readability_index(text),
        "Dale-Chall Readability Score": textstat.dale_chall_readability_score(text),
        "Difficult Words": textstat.difficult_words(text),
        "Linsear Write Formula": textstat.linsear_write_formula(text),
        "Gunning Fog": textstat.gunning_fog(text),
        "Text Standard": textstat.text_standard(text)
    }

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to perform dependency parsing
def dependency_parse(text):
    doc = nlp(text)
    return doc
