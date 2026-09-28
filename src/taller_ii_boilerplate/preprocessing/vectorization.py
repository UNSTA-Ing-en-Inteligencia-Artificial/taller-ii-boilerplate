from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

NEGATIONS = {"no", "not", "nor", "never", "cannot", "nothing", "neither", "none", "nobody", "don't"} # TODO Son algunas, hay que seguir buscando
STOPWORDS = list((set(stopwords.words('english')) - NEGATIONS))

dataset = pd.read_csv("data/preprocessed/dataset.csv")

def bow_vectorizer(documents, n_gram):
    bow = CountVectorizer(ngram_range=n_gram, stop_words=STOPWORDS, analyzer="word")
    result = bow.fit_transform(documents)
    return pd.DataFrame(result.toarray(), columns=bow.get_feature_names_out())


def tfidf_vectorizer(documents, n_gram):
    tfidf = TfidfVectorizer(ngram_range=n_gram, stop_words=STOPWORDS)
    result = tfidf.fit_transform(documents)
    return pd.DataFrame(result.toarray(), columns=tfidf.get_feature_names_out())

bow_result = bow_vectorizer(dataset["review_clean"], (1,1))
bow_result.to_csv("data/final/dataset_bow_unigram.csv", index=False)
bow_result = bow_vectorizer(dataset["review_clean"], (1,2))
bow_result.to_csv("data/final/dataset_bow_unigram_bigram.csv", index=False)
tfidf_result = tfidf_vectorizer(dataset["review_clean"], (1,1))
tfidf_result.to_csv("data/final/dataset_tfidf_unigram.csv", index=False)



print(bow_result.head())
print(tfidf_result.head())