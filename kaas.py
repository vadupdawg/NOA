import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import openai
import os

# Lees het document van bestand
with open('text.txt', 'r', encoding='utf-8') as bestand:
    document = bestand.read()

def summarize(text, per):
    nlp = spacy.load('nl_core_news_lg')
    doc = nlp(text)
    tokens =[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length = max(1, int(len(sentence_tokens) * per))
    # Verkrijg de top N zinnen op basis van score
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

    # Sorteer de geselecteerde zinnen op basis van hun positie in de oorspronkelijke tekst
    ordered_summary = sorted(summary, key=lambda s: sentence_tokens.index(s))

    # Samenvoegen van zinnen tot een enkele tekst
    final_summary = ' '.join([sent.text for sent in ordered_summary])

    print(final_summary)
    return final_summary

def extract_topics_with_gpt4(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4", 
            messages=[
                {"role": "system", "content": "Identificeer de grote, overkoepelende, onderwerpen in de volgende tekst voor als kopjes in een notule, benoem echt alleen de onderwerpen gescheiden door een kommateken:"},
                {"role": "user", "content": text},
            ]
        )
        
        # Krijg het laatste bericht in de response, dat zou het antwoord van de AI moeten zijn.
        topics = response['choices'][0]['message']['content']  # Pas dit pad aan indien nodig
        return topics
    except openai.error.OpenAIError as e:
        return f"Er is een fout opgetreden: {str(e)}"

text = summarize(document, 0.05)
onderwerpen = extract_topics_with_gpt4(text)
print(onderwerpen)

