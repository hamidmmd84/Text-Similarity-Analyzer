from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("TEXT EMBEDDING & SEMANTIC SIMILARITY ANALYZER")
print("=" * 60)


sentences = [
    "هوش مصنوعی و علوم کامپوتر آینده تکنولوژی را تغییر می دهند",
    "یادگیری ماشین و الگوریتم های هوش مصنوعی در حال پیشرفت است",
    "امروز هوا بارانی و سرد است",
    "برنامه نویسی بسیار جذاب است",
]


tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(sentences).toarray()

w2v_vec = CountVectorizer(ngram_range=(1, 2))
X_w2v = w2v_vec.fit_transform(sentences).toarray()

binary_vec = CountVectorizer(binary=True)
X_binary = binary_vec.fit_transform(sentences).toarray()


print("Embedding matrices successfully created.")
print(f"TF-IDF Matrix Shape: {X_tfidf.shape}")
print(f"Word2Vec Matrix Shape: {X_w2v.shape}")
print(f"Binary Matrix Shape: {X_binary.shape}")
print("=" * 60)


svd = TruncatedSVD(n_components=2, random_state=42)
X_tfidf_reduced = svd.fit_transform(X_tfidf)
X_w2v_reduced = svd.fit_transform(X_w2v)
X_binary_reduced = svd.fit_transform(X_binary)

print("A) Dimensionality Reduction With SVD Completed.")
print(f"Reduced TF-IDF Shape: {X_tfidf_reduced.shape}")
print(f"Reduced Word2Vec Shape: {X_w2v_reduced.shape}")
print(f"Reduced Binary Shape: {X_binary_reduced.shape}")
print("=" * 60)


print("B) Calculating Semantic Similarity Between Sentence 1 and Sentence 2:")

sim_tfidf = cosine_similarity([X_tfidf[0]], [X_tfidf[1]])[0][0]
sim_w2v = cosine_similarity([X_w2v[0]], [X_w2v[1]])[0][0]
sim_binary = cosine_similarity([X_binary[0]], [X_binary[1]])[0][0]

print(f"Similarity (TF-IDf): {sim_tfidf:.4f}")
print(f"Similarity (Word2Vec): {sim_w2v:.4f}")
print(f"Similarity (Binary): {sim_binary:.4f}")
print("=" * 60)