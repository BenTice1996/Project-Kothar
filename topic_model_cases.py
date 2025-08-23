import os
import re
import time
import json
import nltk
import pickle
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel


def preprocess(doc, stop_words, lemmatizer):
    tokens = re.findall(r"\b\w+\b", doc.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens


if __name__ == "__main__":
    # === Download NLTK data === #
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    # === USER CONFIGURATION === #
    metadata_path = "" #Insert your file path here
    output_dir = "" #Insert your file path here
    data_dir = "" #Insert your file path here
    topic_range = [10, 20, 30, 40]
    alpha_vals = ['symmetric', 'asymmetric', 0.01, 0.1, 0.5]
    beta_vals = ['symmetric', 0.01, 0.1, 0.5]
    passes = 10
    top_n_terms = 10
    similarity_threshold = 0.7
    random_state = 42

    # === PATH SETUP === #
    top1_out = os.path.join(output_dir, "top_cluster_per_doc.csv")
    top3_out = os.path.join(output_dir, "top_3_clusters_per_doc.csv")
    summary_out = os.path.join(output_dir, "cluster_summaries.json")
    merged_output_path = os.path.join(output_dir, "opinions_with_topics.csv")
    coherence_out_path = os.path.join(output_dir, "coherence_grid_results.csv")
    per_topic_out_path = os.path.join(output_dir, "per_topic_coherence.csv")
    cluster_coherence_out_path = os.path.join(output_dir, "cluster_coherence.csv")

    os.makedirs(output_dir, exist_ok=True)

    # === Load opinion text === #
    print("→ Loading & preprocessing opinions …")
    meta_df = pd.read_csv(metadata_path)
    raw_docs = meta_df["opinion_text"].fillna("").tolist()

    custom_stopwords = set([
        "court", "law", "plaintiff", "defendant", "see", "act", "f3d", "district", "state", "cir", "case",
        "petitioner", "appellee", "appellant", "plaintiffs", "defendants", "petitioners", "appellees", "appellants",
        "respondent", "respondents", "united", "federal", "party", "counsel", "rptr", "lawyer", "infra", "supra", "id", "supp", "f2d", "fd"
    ])
    stop_words = set(stopwords.words("english")) | custom_stopwords
    lemmatizer = WordNetLemmatizer()

    tokenized_docs = [preprocess(doc, stop_words, lemmatizer) for doc in raw_docs]

    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

    # === Save Preprocessed Data === #
    joblib.dump(tokenized_docs, os.path.join(data_dir, "tokenized_docs.pkl"))
    joblib.dump(dictionary, os.path.join(data_dir, "dictionary.pkl"))
    joblib.dump(corpus, os.path.join(data_dir, "corpus.pkl"))

    # === GRID SEARCH FOR BEST LDA MODEL === #
    print("\n→ Running grid search …")
    grid_results = []
    topic_results = {}
    lda_models = {}

    for num_topics in topic_range:
        for alpha in alpha_vals:
            for beta in beta_vals:
                label = f"K{num_topics}_alpha{alpha}_beta{beta}"
                print(f"   ⤷ training {label} …", flush=True)

                lda = models.LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    passes=passes,
                    random_state=random_state,
                    alpha=alpha,
                    eta=beta
                )

                coherence_model = CoherenceModel(
                    model=lda,
                    texts=tokenized_docs,
                    dictionary=dictionary,
                    coherence='c_v',
                    processes=1  # ❗ FIX for Windows multiprocessing
                )
                coherence = coherence_model.get_coherence()
                per_topic = coherence_model.get_coherence_per_topic()

                grid_results.append({
                    "label": label,
                    "num_topics": num_topics,
                    "alpha": alpha,
                    "beta": beta,
                    "coherence": coherence
                })

                topic_results[label] = per_topic
                lda_models[label] = lda
                lda.save(os.path.join(output_dir, f"{label}.lda"))

    pd.DataFrame(grid_results).to_csv(coherence_out_path, index=False)

    topic_rows = []
    for label, scores in topic_results.items():
        for topic_id, score in enumerate(scores):
            topic_rows.append({
                "label": label,
                "topic_id": topic_id,
                "topic_coherence": score
            })

    pd.DataFrame(topic_rows).to_csv(per_topic_out_path, index=False)

    # === Select Best Model === #
    best_model_entry = max(grid_results, key=lambda x: x['coherence'])
    best_label = best_model_entry['label']
    best_lda = lda_models[best_label]
    best_num_topics = best_model_entry['num_topics']
    print(f"\n→ Best model = {best_label} (coherence = {best_model_entry['coherence']:.4f})")

    # === CLUSTER TOPICS OF BEST MODEL === #
    print("→ Clustering topics …")
    topic_term_matrix = best_lda.get_topics()
    cos_sim_matrix = cosine_similarity(topic_term_matrix)

    clustering = AgglomerativeClustering(
        metric='cosine', linkage='average',
        distance_threshold=1 - similarity_threshold,
        n_clusters=None
    ).fit(topic_term_matrix)

    clusters = defaultdict(list)
    for topic_id, cluster_id in enumerate(clustering.labels_):
        clusters[cluster_id].append(topic_id)

    best_topic_df = pd.read_csv(per_topic_out_path)
    best_topic_df = best_topic_df[best_topic_df["label"] == best_label]

    cluster_records = []
    cluster_lookup = {}

    for cluster_id, topic_ids in clusters.items():
        term_scores = np.mean(topic_term_matrix[topic_ids], axis=0)
        top_term_ids = term_scores.argsort()[-top_n_terms:][::-1]
        top_terms = [str(dictionary[int(i)]) for i in top_term_ids]
        coherence_vals = best_topic_df.loc[best_topic_df["topic_id"].isin(topic_ids), "topic_coherence"]
        avg_coherence = float(np.mean(coherence_vals))

        cluster_records.append({
            "cluster_id": cluster_id,
            "num_topics": len(topic_ids),
            "average_topic_coherence": avg_coherence,
            "top_terms": "; ".join(top_terms)
        })

        cluster_lookup[cluster_id] = ", ".join(top_terms)

    pd.DataFrame(cluster_records).to_csv(cluster_coherence_out_path, index=False)
    
    # Convert all NumPy types to plain Python ints/floats
    for record in cluster_records:
        record["cluster_id"] = int(record["cluster_id"])
        record["num_topics"] = int(record["num_topics"])
        record["average_topic_coherence"] = float(record["average_topic_coherence"])
        # top_terms is already a string

    with open(summary_out, "w") as f:
        json.dump(cluster_records, f, indent=2)


    # === DOCUMENT-CLUSTER SCORING === #
    print("→ Scoring documents …")
    doc_cluster_scores = defaultdict(lambda: defaultdict(float))

    for doc_id, tokens in enumerate(tokenized_docs):
        doc_bow = dictionary.doc2bow(tokens)
        if not doc_bow:
            continue
        for topic_id, prob in best_lda.get_document_topics(doc_bow, minimum_probability=0.0):
            cluster_id = clustering.labels_[topic_id]
            doc_cluster_scores[doc_id][cluster_id] += prob

    records = []
    for doc_id, scores in doc_cluster_scores.items():
        total = sum(scores.values()) or 1.0
        for cluster_id, score in scores.items():
            records.append({
                "doc_id": doc_id,
                "cluster_id": cluster_id,
                "score": score / total
            })

    doc_cluster_df = pd.DataFrame(records)
    print(f"Scoring complete: {len(doc_cluster_df)} doc-cluster rows")

    # === EXPORT TOP 1 CLUSTER === #
    top_cluster_df = doc_cluster_df.sort_values("score", ascending=False).groupby("doc_id").head(1)
    top_cluster_df = top_cluster_df.rename(columns={"cluster_id": "top_cluster", "score": "top_score"})
    top_cluster_df.to_csv(top1_out, index=False)

    # === EXPORT TOP 3 CLUSTERS === #
    top_n_df = doc_cluster_df.sort_values("score", ascending=False).groupby("doc_id").head(3)
    top_n_df = top_n_df.rename(columns={"cluster_id": "related_cluster", "score": "relevance_score"})
    top_n_df.to_csv(top3_out, index=False)

    # === ENRICH METADATA WITH TOPIC LABELS === #
    print("→ Enriching metadata with cluster terms …")
    meta_df["doc_id"] = meta_df.index
    merged = meta_df.merge(top_cluster_df, on="doc_id", how="left")
    merged["top_cluster_terms"] = merged["top_cluster"].map(cluster_lookup)

    top3_df = pd.read_csv(top3_out)
    top3_df["top_terms"] = top3_df["related_cluster"].map(cluster_lookup)
    top3_summary = top3_df.groupby("doc_id").apply(
        lambda df: "; ".join(
            f"#{row.related_cluster} ({row.relevance_score:.2f}): {row.top_terms}"
            for _, row in df.iterrows()
        )
    ).reset_index(name="top_3_clusters_summary")

    merged = merged.merge(top3_summary, on="doc_id", how="left")
    merged.to_csv(merged_output_path, index=False)

    print("\n✅ Script complete! All outputs saved to:")
    print(f"  - {coherence_out_path}")
    print(f"  - {per_topic_out_path}")
    print(f"  - {cluster_coherence_out_path}")
    print(f"  - {top1_out}")
    print(f"  - {top3_out}")
    print(f"  - {merged_output_path}")

