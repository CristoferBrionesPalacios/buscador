import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
import requests
from collections import defaultdict, Counter
import operator
import langid

# Descargar los recursos necesarios para nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')

def get_page_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener la página {url}: {e}")
        return ""

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    text = soup.get_text()
    return text

def process_url(url, stopwords_set, nltk_words):
    content = get_page_content(url)
    if content:
        print(f"Procesando {url}")
        text = extract_text_from_html(content)
        words = nltk.word_tokenize(text)
        filtered_words = [
            word.lower() for word in words
            if word.isalpha() and
            word.lower() not in stopwords_set and
            word.lower() in nltk_words and
            langid.classify(word)[0] == 'es'
        ]
        return Counter(filtered_words)
    else:
        print(f"Contenido de la página {url} no disponible.")
        return Counter()

def build_inverted_index(urls, nltk_words):
    inverted_index = defaultdict(Counter)
    stopwords_set = set(stopwords.words('spanish'))

    for url in urls:
        word_counter = process_url(url, stopwords_set, nltk_words)
        inverted_index[url] = word_counter

    total_counter = sum((Counter(dict(counter)) for counter in inverted_index.values()), Counter())

    inverted_index_final = defaultdict(list)
    for url, word_counter in inverted_index.items():
        for word, frequency in word_counter.items():
            inverted_index_final[word].append((url, frequency))

    return inverted_index_final

def save_inverted_index(inverted_index, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, frequencies in sorted(inverted_index.items(), key=lambda x: x[0]):
            sorted_frequencies = sorted(frequencies, key=operator.itemgetter(1), reverse=True)
            f.write(f"{word}: {sorted_frequencies}\n")

def main():
    input_file = 'urls.txt'
    output_file = 'raiz_ind_inv.txt'

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        urls = [line.strip() for line in f.readlines()]

    nltk_words = set(stopwords.words('spanish') + nltk.corpus.words.words())
    
    inverted_index = build_inverted_index(urls, nltk_words)
    save_inverted_index(inverted_index, output_file)

if __name__ == "__main__":
    main()
