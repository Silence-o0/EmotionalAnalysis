import pandas as pd
import re
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()
nrc_file_path = 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
target_emotions = {'fear', 'negative'}


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_text(text):
    words_in_text = re.findall(r'\b\w+\b', text.lower())
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words_in_text]
    return lemmatized_words


def load_nrc_lexicon(filepath):
    lexicon = {}

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                word, emotion, association = parts
                if emotion in target_emotions and int(association) == 1:
                    if word not in lexicon:
                        lexicon[word] = []
                    lexicon[word].append(emotion)

    return lexicon


def count_panic_words(text, word_list):
    word_count = Counter()
    lemmatized_words = lemmatize_text(text)
    for word in word_list:
        word_count[word] = lemmatized_words.count(word)
    return word_count


def bootstrapping_word_frequencies(df, word_list, num_iterations=20):
    word_freqs = Counter()

    for i in range(num_iterations):
        sample_df = df.sample(frac=1, replace=True)

        for text in sample_df['text']:
            word_counts = count_panic_words(text, word_list)
            word_freqs.update(word_counts)

    total_count = sum(word_freqs.values())
    word_weights = {word: freq / total_count for word, freq in word_freqs.items()}

    return word_weights

def evaluate_article(text, word_weights):
    word_count = count_panic_words(text, word_weights.keys())
    score = sum(word_weights[word] * count for word, count in word_count.items())

    print("\nFounded panic words and their quantity:")
    for word, count in word_count.items():
        if count > 0:
            print(f"Word: {word}, Quantity: {count}, Weight: {word_weights[word]}")

    normalized_score = score / len(re.findall(r'\b\w+\b', text.lower()))
    return normalized_score


if __name__ == '__main__':
    panic_words = load_nrc_lexicon(nrc_file_path)
    panic_word_list = list(panic_words.keys())

    df = pd.read_excel('news_api_dataset.xlsx')
    word_weights = bootstrapping_word_frequencies(df, panic_word_list, num_iterations=10)

    test_article = """
    A massive earthquake has caused catastrophic damage to the city. Buildings have been destroyed, roads are flooded, and people are left without food and water.
    The death toll is rising by the hour. Rescue operations are ongoing, but the situation remains critical.
    Local authorities are warning of possible further landslides and other natural disasters. People are urged to evacuate the danger zone immediately.
    """

    score = evaluate_article(test_article, word_weights)
    print(f"\nArticle panic estimation: {score}")
