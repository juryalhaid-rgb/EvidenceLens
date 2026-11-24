"""
EvidenceLens - core IR functions (DS321 Information Retrieval project).

Contains: corpus cleaning, text preprocessing (normalization, stopword
removal, stemming), PyTerrier index construction, TF-IDF / BM25 retrievers,
and the pooling-sheet generator used for manual relevance judgments.
"""

import os
import re
import unicodedata

import pandas as pd

MIN_ABSTRACT_WORDS = 30          # abstracts shorter than this are not informative
META_FIELDS = ["docno", "title", "abstract", "url", "year", "journal"]

# --------------------------------------------------------------------------
# 1. Corpus cleaning
# --------------------------------------------------------------------------
def clean_corpus(df):
    """Clean the raw PubMed table. Returns (clean_df, report_dict)."""
    report = {"raw_documents": len(df)}
    df = df.copy()
    for col in ["docno", "title", "abstract"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # (a) missing values
    before = len(df)
    df = df[(df.docno != "") & (df.title != "") & (df.abstract != "")]
    report["dropped_missing_values"] = before - len(df)

    # (b) duplicate document ids
    before = len(df)
    df = df.drop_duplicates(subset="docno", keep="first")
    report["dropped_duplicate_docno"] = before - len(df)

    # (c) too-short / placeholder abstracts, e.g. "[Figure: see text]."
    before = len(df)
    df = df[df.abstract.str.split().str.len() >= MIN_ABSTRACT_WORDS]
    report["dropped_short_abstracts"] = before - len(df)

    # (d) duplicate abstracts (same text under different PMIDs)
    before = len(df)
    df = df.drop_duplicates(subset="abstract", keep="first")
    report["dropped_duplicate_abstracts"] = before - len(df)

    df = df.reset_index(drop=True)
    report["final_documents"] = len(df)
    return df, report


# --------------------------------------------------------------------------
# 2. Preprocessing: normalization -> stopword removal -> stemming
# --------------------------------------------------------------------------
_STOPWORDS = None
_STEMMER = None
_STEM_CACHE = {}


def _init_nlp():
    """Load NLTK resources once (lazy, so the module imports without NLTK)."""
    global _STOPWORDS, _STEMMER
    if _STEMMER is None:
        import nltk
        from nltk.stem import PorterStemmer
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        _STOPWORDS = set(stopwords.words("english"))
        _STEMMER = PorterStemmer()


def normalize(text):
    """Lowercase, unify unicode, replace punctuation/hyphens with spaces,
    and drop tokens that are only digits or single characters."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # keeps letters/digits only
    return [t for t in text.split() if len(t) > 1 and not t.isdigit()]


def preprocess(text):
    """Full pipeline used for BOTH documents and queries. Returns a string
    of space-separated stemmed tokens."""
    _init_nlp()
    tokens = [t for t in normalize(text) if t not in _STOPWORDS]
    stems = []
    for t in tokens:
        if t not in _STEM_CACHE:
            _STEM_CACHE[t] = _STEMMER.stem(t)
        stems.append(_STEM_CACHE[t])
    return " ".join(stems)


# --------------------------------------------------------------------------
# 3. Inverted index (PyTerrier)
# --------------------------------------------------------------------------

def _start_pyterrier():
    """Import PyTerrier and start Java in a way that works across versions
    (older versions need pt.init(); newer ones start Java automatically)."""
    import pyterrier as pt
    try:
        if hasattr(pt, "started") and not pt.started():
            pt.init()
    except Exception:
        pass
    return pt

def build_index(df, index_dir="index"):
    """Build a Terrier inverted index from already-cleaned documents.
    Text is preprocessed by OUR preprocess(); Terrier's own stemmer and
    stopword list are switched off to avoid processing twice. Original
    title/abstract/url are stored as metadata for display in the UI."""
    pt = _start_pyterrier()

    def doc_iter():
        for row in df.itertuples(index=False):
            yield {
                "docno": str(row.docno),
                "text": preprocess(f"{row.title} {row.abstract}"),
                "title": row.title,
                "abstract": row.abstract,
                "url": row.url,
                "year": str(row.year),
                "journal": row.journal,
            }

    indexer = pt.IterDictIndexer(
        os.path.abspath(index_dir),
        meta={"docno": 20, "title": 512, "abstract": 8192,
              "url": 64, "year": 8, "journal": 256},
        overwrite=True,
        stemmer=None,      # we already stemmed
        stopwords=None,    # we already removed stopwords
    )
    index_ref = indexer.index(doc_iter())
    return pt.IndexFactory.of(index_ref)


def load_index(index_dir="index"):
    pt = _start_pyterrier()
    return pt.IndexFactory.of(os.path.abspath(index_dir))


# --------------------------------------------------------------------------
# 4. Retrieval models (BatchRetrieve, top 10)
# --------------------------------------------------------------------------
def get_retrievers(index, num_results=10):
    """TF-IDF and BM25 over the SAME index. Query-time term pipelines are
    emptied because queries are preprocessed by preprocess()."""
    pt = _start_pyterrier()
    # The course requires BatchRetrieve; newer PyTerrier keeps it as a
    # (deprecated) alias of pt.terrier.Retriever, so fall back if missing.
    Retr = getattr(pt, "BatchRetrieve", None) or pt.terrier.Retriever
    common = dict(num_results=num_results,
                  metadata=["docno", "title", "abstract", "url", "year", "journal"],
                  properties={"termpipelines": ""})
    return {
        "TF-IDF": Retr(index, wmodel="TF_IDF", **common),
        "BM25": Retr(index, wmodel="BM25", **common),
    }


def search(retriever, raw_query):
    """Preprocess one raw query and return the ranked top-10 DataFrame
    (docid, docno, rank, score, + metadata)."""
    processed = preprocess(raw_query)
    if not processed.strip():
        return pd.DataFrame()
    return retriever.transform(pd.DataFrame([{"qid": "1", "query": processed}]))


# --------------------------------------------------------------------------
# 5. Topics + pooling sheet for manual judgments
# --------------------------------------------------------------------------
def load_topics(path="topics.csv"):
    """Read topics (num, title, desc) and add PyTerrier columns qid/query.
    The query is the preprocessed TITLE."""
    topics = pd.read_csv(path, dtype=str)
    topics["qid"] = topics["num"]
    topics["query"] = topics["title"].apply(preprocess)
    return topics


def make_pooling_sheet(topics, retrievers, corpus, n_random=5, seed=42):
    """Pool top-10 of each model per topic + a few random documents, shuffle,
    and hide which system retrieved what (reduces judging bias).
    Returns (judging_sheet, sources_map)."""
    import random
    rng = random.Random(seed)
    all_docnos = corpus.docno.tolist()
    by_docno = corpus.set_index("docno")
    rows, sources = [], []

    for t in topics.itertuples(index=False):
        pool = {}
        for name, retr in retrievers.items():
            res = retr.transform(pd.DataFrame([{"qid": t.qid, "query": t.query}]))
            for d in res.docno:
                pool.setdefault(d, set()).add(name)
        pooled = set(pool)
        randoms = []
        while len(randoms) < n_random:
            d = rng.choice(all_docnos)
            if d not in pooled and d not in randoms:
                randoms.append(d)
        for d in randoms:
            pool[d] = {"random"}

        docs = list(pool)
        rng.shuffle(docs)
        for d in docs:
            r = by_docno.loc[d]
            rows.append({"qid": t.qid, "topic": t.title, "topic_description": t.desc,
                         "docno": d, "title": r.title, "abstract": r.abstract,
                         "url": r.url, "label": ""})
            sources.append({"qid": t.qid, "docno": d, "found_by": "+".join(sorted(pool[d]))})
    return pd.DataFrame(rows), pd.DataFrame(sources)
