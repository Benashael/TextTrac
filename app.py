import streamlit as st
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import FreqDist
import textstat
import re
import contractions
import base64

# Set up Streamlit app
st.set_page_config(page_title="TextTrac", page_icon="✍️", layout="wide")

st.title("TextTrac 📊✍️: Navigate Text Data with AutoNLP")

page = st.sidebar.radio("**🌐 Select a Feature**", ["Home Page 🏠", "Text Statistics 📊", "Tokenization 🔠", "Stopwords Removal 🛑", "POS Tagging 🏷️", "Stemming 🌱", "Lemmatization 🌿", "Text Normalization 🧮", "N-Grams 🔢", "Keyword Extraction 🔑", "Synonym and Antonym Detection 🔤", "Text Similarity 🔄", "Text Complexity Analysis 📊", "Word Cloud ☁️"])

def clear_session_state():
    st.session_state.pop("input_type", None)
    st.session_state.pop("text_input", None)
    st.session_state.pop("uploaded_file", None)
    st.session_state.pop("input_data", None)
    st.session_state.pop("max_word_limit", None)

def get_input():
    if "input_type" not in st.session_state:
        st.session_state.input_type = "Text Input"

    input_type = st.radio("**🔍 Choose input type**", ["Text Input", "TXT File Upload", "Example Dataset"], key="input_type")

    if input_type == "Text Input":
        max_word_limit = 300
        st.write(f"⚠️ Maximum Word Limit: {max_word_limit} words")
        text_input = st.text_area("📝 Enter text:", key="text_input")
        if st.button("📝 Submit Text"):
            if not text_input.strip():
                st.error("❌ Error: Text input cannot be blank.")
            else:
                word_count = len(text_input.split())
                if word_count > max_word_limit:
                    st.error(f"❌ Word count exceeds the maximum limit of {max_word_limit} words.")
                else:
                    st.session_state.input_data = text_input
                    st.session_state.max_word_limit = max_word_limit

    elif input_type == "TXT File Upload":
        max_word_limit = 3000
        st.write(f"⚠️ Maximum Word Limit: {max_word_limit} words")
        uploaded_file = st.file_uploader("📄 Upload a text file", type=["txt"], key="uploaded_file")
        if st.button("📄 Submit File"):
            if uploaded_file is not None:
                try:
                    file_contents = uploaded_file.read().decode("utf-8")
                    if not file_contents.strip():
                        st.error("❌ Error: The uploaded file is empty.")
                    else:
                        word_count = len(text_input.split())
                        if word_count > max_word_limit:
                            st.error(f"❌ Word count exceeds the maximum limit of {max_word_limit} words.")
                        else:
                            st.session_state.input_data = file_contents
                            st.session_state.max_word_limit = max_word_limit
                except UnicodeDecodeError:
                    st.error("❌ Error: The uploaded file contains non-text data or is not in UTF-8 format.")
            else:
                st.error("❌ Error: Please upload a file.")
    
    elif input_type == "Example Dataset":
        example_dataset = "example_dataset.txt"
        with open('example_dataset.txt', 'r') as file:
            lines = file.readlines()
        for line in lines:
            file_contents = line.strip()
            st.session_state.input_data = file_contents
        
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
    st.subheader("🔑 Keywords and their Frequencies (Dataframe📊💼):")
    st.dataframe(keywords_df)
    
    # Plot keyword frequency distribution
    st.subheader("🔑 Keywords and their Frequencies (Visualization Plot📈🎨):")
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
    filtered_tokens = remove_stopwords(tokens)
    results = []
    for token in filtered_tokens:
        if token.isalnum():
            synonyms, antonyms = find_synonyms_antonyms(token)
            results.append((token, synonyms, antonyms))
    return results

# Function to perform text complexity analysis
def analyze_text_complexity(text):
    return {
        "**😊 Flesch Reading Ease**": {
            "Score": f"{textstat.flesch_reading_ease(text)} out of 100",
            "Explanation": "Measures the ease of readability on a scale from 0 to 100, with higher scores indicating easier readability."
        },
        "**🧠 Smog Index**": {
            "Score": f"{textstat.smog_index(text)} out of 30",
            "Explanation": "Estimates the years of education a person needs to comprehend the text on first reading."
        },
        "**📚 Flesch-Kincaid Grade**": {
            "Score": textstat.flesch_kincaid_grade(text),
            "Explanation": "Indicates the grade level needed to understand the text, with a higher grade indicating more complexity."
        },
        "**🤔 Coleman-Liau Index**": {
            "Score": textstat.coleman_liau_index(text),
            "Explanation": "Computes the approximate U.S. grade level needed to comprehend the text."
        },
        "**📖 Automated Readability Index**": {
            "Score": textstat.automated_readability_index(text),
            "Explanation": "Estimates the U.S. grade level needed to understand the text, with higher values indicating more advanced readability."
        },
        "**📝 Dale-Chall Readability Score**": {
            "Score": textstat.dale_chall_readability_score(text),
            "Explanation": "Evaluates the comprehension difficulty of text, considering a list of familiar words."
        },
        "**📚🔍 Difficult Words**": {
            "Score": f"{textstat.difficult_words(text)} out of total words",
            "Explanation": "Counts the number of difficult words within the text, providing insight into its complexity."
        },
        "**🧐 Linsear Write Formula**": {
            "Score": textstat.linsear_write_formula(text),
            "Explanation": "Estimates the readability of English text by looking at the number of simple and complex words."
        },
        "**🤓 Gunning Fog**": {
            "Score": f"{textstat.gunning_fog(text)} out of 20",
            "Explanation": "Measures the readability of English writing, considering sentence length and the complexity of words."
        },
        "**🎓 Text Standard**": {
            "Score": textstat.text_standard(text),
            "Explanation": "Indicates the U.S. grade level for which the text is most suitable."
        }
    }

# Function to perform text normalization
def normalize_text(text):
    # Lowercase text
    text = text.lower()
    
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Remove extra whitespace
    text = ' '.join(tokens)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Character Count Function
def count_characters(text):
    return len(text)

# Word Count Function
def count_words(text):
    return len(text.split())

# Sentence Splitting Function
def count_sentences(text):
    sentences = sent_tokenize(text)
    return len(sentences)

def download_button(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">⬇️ Download Normalized Text</a>'
    st.markdown(href, unsafe_allow_html=True)

# List of pages to exclude the common input section
exclude_input_pages = ["Home Page 🏠", "Text Similarity 🔄"]

if page not in exclude_input_pages:
    # Common input section for pages not in the exclude list
    get_input()

    # Add a button to clear the session state
    if st.button("🗑️ Clear Input"):
        clear_session_state()
        st.experimental_rerun()

    st.info("⚠️ Click '🗑️ Clear Input' to reset the text input and file upload fields. This will clear all entered data and allow you to start fresh.")

# Page 1
if page == "Home Page 🏠":
    st.subheader("Explore the power of text manipulation and analysis with cutting-edge tools and techniques.")
    st.markdown("___________")
    st.header("Navigate through the following features:")
    st.markdown("""
    ✨ **Text Statistics 📊:** Calculate the number of characters, words, and sentences in the text.
    
    ✨ **Text Normalization 🧮:** Preprocess the text to ensure consistency (e.g., lowercasing, removing punctuation).
    
    ✨ **Tokenization 🔠:** Break down text into its individual components for deeper analysis.

    ✨ **Stopwords Removal 🛑:** Cleanse your text of common words to focus on the most meaningful content.
    
    ✨ **POS Tagging 🏷️:** Understand the grammatical structure of your text with part-of-speech tagging.
    
    ✨ **Stemming 🌱:** Simplify words to their root form for streamlined analysis.
    
    ✨ **Lemmatization 🌿:** Transform words to their base or dictionary form for accurate analysis.
    
    ✨ **N-Grams 🔢:** Explore sequences of words for deeper insights into your text's structure.
    
    ✨ **Keyword Extraction 🔑:** Identify the most important terms in your text for efficient analysis.
    
    ✨ **Synonym and Antonym Detection 🔤:** Discover alternative words and their opposites to enrich your text.
    
    ✨ **Text Similarity 🔄:** Measure the likeness between texts to identify similarities and differences.
    
    ✨ **Text Complexity Analysis 📊:** Assess the complexity of your text to tailor your analysis approach.

    ✨ **Word Cloud ☁️:** Visualize the most frequent words in your text with beautiful word clouds.
    """)
   
# Page 2
elif page == "Text Statistics 📊":
    st.header("📊 Text Statistics Feature")

    if "input_data" in st.session_state:
        if st.button("👀 Show Statistics"):
            char_count = count_characters(st.session_state.input_data)
            st.subheader("📏 Character Count:")
            st.write(char_count)
            word_count = count_words(st.session_state.input_data)
            st.subheader("🧮 Word Count:")
            st.write(word_count)
            sent_count = count_sentences(st.session_state.input_data)
            st.subheader("🗒️ Word Count:")
            st.write(sent_count)
    else:
        st.info("⚠️ Please provide text input, upload a file, or use an example dataset.")

# Page 3
elif page == "Text Normalization 🧮":
    st.title("🧮 Text Normalization Feature")

    if "input_data" in st.session_state:
        if st.button("🔍 Normalize Text"):
            normalized_text = normalize_text(st.session_state.input_data)
            st.subheader("🔍 Normalized Text:")
            st.markdown(f'<div style="background-color:#F0F0F0; padding:10px; border-radius:5px">{normalized_text}</div>', unsafe_allow_html=True)
            download_button(normalized_text, "normalized_text.txt")
    else:
        st.info("⚠️ Please provide text input, upload a file, or use an example dataset.")

# Page 2
elif page == "Tokenization 🔠":
    st.header("🔠 Tokenization Feature")
    
    if "input_data" in st.session_state:
        tokenization_type = st.radio("**🧩 Choose tokenization type**", ["Word Tokenization", "Sentence Tokenization"])
        if st.button("🚀 Perform Tokenization"):
            tokens = tokenize_text(st.session_state.input_data, tokenization_type)
            st.subheader("🔍 Tokens:")
            st.write(tokens)
    else:
        st.info("⚠️ Please provide text input, upload a file, or use an example dataset.")

# Page 3
elif page == "POS Tagging 🏷️":
    st.header("🏷️ POS Tagging Feature")
    tokenization_type = "Word Tokenization"

    if "input_data" in st.session_state:
        if st.button("🚀 Perform POS Tagging"):
            tokens = tokenize_text(st.session_state.input_data, tokenization_type)
            pos_tags = pos_tagging(tokens)
            pos_emoji_desc = {
                'NN': ('📝', 'Noun'),
                'NNS': ('📝', 'Nouns (plural)'),
                'VB': ('🔧', 'Verb (base form)'),
                'VBD': ('🔧', 'Verb (past tense)'),
                'VBG': ('🔧', 'Verb (gerund/present participle)'),
                'VBN': ('🔧', 'Verb (past participle)'),
                'VBP': ('🔧', 'Verb (non-3rd person singular present)'),
                'VBZ': ('🔧', 'Verb (3rd person singular present)'),
                'JJ': ('✨', 'Adjective'),
                'RB': ('🌀', 'Adverb'),
                'IN': ('🔗', 'Preposition/subordinating conjunction'),
                'DT': ('🔠', 'Determiner'),
                'PRP': ('🙋', 'Personal pronoun'),
                'PRP$': ('🙋', 'Possessive pronoun'),
                'CC': ('🔗', 'Coordinating conjunction'),
                'UH': ('😲', 'Interjection'),
                'TO': ('➡️', 'to'),
                'MD': ('🛠️', 'Modal')
            }
            pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
            pos_df['Icon'] = pos_df['POS Tag'].apply(lambda tag: pos_emoji_desc.get(tag, ('❓', 'Unknown'))[0])
            pos_df['Description'] = pos_df['POS Tag'].apply(lambda tag: pos_emoji_desc.get(tag, ('❓', 'Unknown'))[1])
            st.subheader("🏷️ POS Tags with Icons and Descriptions::")
            st.dataframe(pos_df)
    else:
        st.info("⚠️ Please provide text input, upload a file, or use an example dataset.")

# Page 4
elif page == "Stopwords Removal 🛑":
    st.header("🛑 Stopwords Removal Feature")
    tokenization_type = "Word Tokenization"

    if "input_data" in st.session_state:
        if st.button("🚫 Remove Stopwords"):
            tokens = tokenize_text(st.session_state.input_data, tokenization_type)
            st.subheader("📝 Tokens (Before Stopwords Removal):")
            st.write(tokens)
            # Remove stopwords
            filtered_tokens = remove_stopwords(tokens)
            st.subheader("🚫 Tokens (After Stopwords Removal):")
            st.write(filtered_tokens)
    else:
        st.info("⚠️ Please provide text input, upload a file, or use an example dataset.")

# Page 5
elif page == "Stemming 🌱":
    st.header("🌱 Stemming Feature")
    tokenization_type = "Word Tokenization"

    if "input_data" in st.session_state:
        if st.button("✂️ Perform Stemming"):
            tokens = tokenize_text(st.session_state.input_data, tokenization_type)
            filtered_tokens = remove_stopwords(tokens)
            st.subheader("🌱 Tokens (Before Stemming):")
            st.write(filtered_tokens)
            # Perform stemming
            stemmed_tokens = perform_stemming(filtered_tokens)
            st.subheader("✂️ Tokens (After Stemming):")
            st.write(stemmed_tokens)
    else:
        st.info("⚠️ Please provide text input, upload a file, or use an example dataset.")  

# Page 6
elif page == "Lemmatization 🌿":
    st.header("🌿 Lemmatization Feature")
    tokenization_type = "Word Tokenization"

    if "input_data" in st.session_state:
        if st.button("📚 Perform Lemmatization"):
            tokens = tokenize_text(st.session_state.input_data, tokenization_type)
            filtered_tokens = remove_stopwords(tokens)
            st.subheader("🌿 Tokens (Before Lemmatization):")
            st.write(filtered_tokens)
            # Perform stemming
            lemmatized_tokens = perform_lemmatization(filtered_tokens)
            st.subheader("📚 Tokens (After Lemmatization):")
            st.write(lemmatized_tokens)
    else:
        st.info("⚠️ Please provide text input, upload a file, or use an example dataset.")  

# Page 7
elif page == "Word Cloud ☁️":
    st.header("☁️ Word Cloud Feature")
    tokenization_type = "Word Tokenization"

    if "input_data" in st.session_state:
        if st.button("⚙️ Generate Word Cloud"):
            tokens = tokenize_text(st.session_state.input_data, tokenization_type)
            filtered_tokens = remove_stopwords(tokens)
            st.subheader("☁️ Word Cloud:")
            generate_word_cloud(filtered_tokens)
    else:
        st.info("⚠️ Please provide text input, upload a file, or use an example dataset.")

# Page 8
elif page == "N-Grams 🔢":
    st.header("🔢 N-Grams Feature")
    tokenization_type = "Word Tokenization"

    if "input_data" in st.session_state:
        n_gram_type = st.radio("**🧩 Choose N-Gram Type**", ["Uni-Grams (1-Grams)", "Bi-Grams (2-Grams)", "Tri-Grams (3-Grams)"])
        if n_gram_type == "Uni-Grams (1-Grams)":
            n = 1
        elif n_gram_type == "Bi-Grams (2-Grams)":
            n = 2
        elif n_gram_type == "Tri-Grams (3-Grams)":
            n = 3
        if st.button("⚙️ Generate N-Grams"):
            tokens = tokenize_text(st.session_state.input_data, tokenization_type)
            n_grams = create_ngrams(tokens, n)
            n_grams_text = generate_ngrams_text(n_grams)    
            st.subheader(f"🛑 {n}-Grams (With Stopwords):")
            st.write(n_grams_text) 
            filtered_tokens = remove_stopwords(tokens)
            n_grams = create_ngrams(filtered_tokens, n)
            n_grams_text = generate_ngrams_text(n_grams)    
            st.subheader(f"🚫 {n}-Grams (Without Stopwords):")
            st.write(n_grams_text) 
    else:
        st.info("⚠️ Please provide text input, upload a file, or use an example dataset.")

# Page 9
elif page == "Keyword Extraction 🔑":
    st.header("🔑 Keyword Extraction Feature")

    if "input_data" in st.session_state:
        if st.button("🔍 Extract Keywords"):
            extract_keywords(st.session_state.input_data)
    else:
        st.info("⚠️ Please provide text input, upload a file, or use an example dataset.")

# Page 10
elif page == "Synonym and Antonym Detection 🔤":
    st.header("🔤 Synonym and Antonym Detection Feature")

    if "input_data" in st.session_state:
        if st.button("🔍 Find Synonyms and Antonyms"):
            results = process_text_for_synonyms_antonyms(st.session_state.input_data)
            results_df = pd.DataFrame(results, columns=["Word", "Synonyms", "Antonyms"])
            st.subheader("🔍 Synonyms and Antonyms:")
            st.dataframe(results_df)
    else:
        st.info("⚠️ Please provide text input, upload a file, or use an example dataset.")

# Page 11
elif page == "Text Similarity 🔄":
    st.header("🔄 Text Similarity Feature")
    max_word_limit = 300
    st.write(f"⚠️ Maximum Word Limit: {max_word_limit} words")
    text1 = st.text_area("📝 Enter text 1:", key="text_input_1")
    text2 = st.text_area("📝 Enter text 2:", key="text_input_2")
    if st.button("🔍 Find Text Similarity"):
        if not text1.strip() or not text2.strip():
            st.error("⚠️ Please provide both texts for similarity comparison.")
        elif len(word_tokenize(text1)) > max_word_limit or len(word_tokenize(text2)) > max_word_limit:
            st.error(f"❌ Word count exceeds the maximum limit of {max_word_limit} words.")
        else:
            similarity_score = calculate_similarity(text1, text2)
            st.subheader("🔄 Similarity Score:")
            st.write(f"**The cosine similarity between the two texts is:** {similarity_score:.2f}")

# Page 12
elif page == "Text Complexity Analysis 📊":
    st.header("📊 Text Complexity Analysis Feature")

    if "input_data" in st.session_state:
        if st.button("🚀 Analyze Text Complexity"):
            st.subheader("📈 Text Complexity Analysis Results:")
            complexity_results = analyze_text_complexity(st.session_state.input_data)
            for metric, data in complexity_results.items():
                st.write(f"{metric}: {data['Score']}")
                st.write(f"Explanation: {data['Explanation']}")
    else:
        st.info("⚠️ Please provide text input, upload a file, or use an example dataset.")
