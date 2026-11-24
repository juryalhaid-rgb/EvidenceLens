"""
Pure text helpers for the EvidenceLens web interface (no Streamlit/Terrier
needed, so they can be tested on their own).

Snippets and highlighting use the SAME preprocess() as the index: a word in
an abstract is highlighted when its stem equals a stem of the query, so what
you see highlighted is exactly what the retrieval model matched on.
"""
import re
from html import escape

_WORD = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def query_stems(query, preprocess):
    """Set of stemmed query terms."""
    return set(preprocess(query).split())


def _matches(word, stems, preprocess):
    stem = preprocess(word)
    return bool(stem) and stem in stems


def highlight(text, stems, preprocess):
    """Return HTML-safe text with matching words wrapped in <mark>."""
    out, pos = [], 0
    for m in _WORD.finditer(text):
        out.append(escape(text[pos:m.start()]))
        word = m.group()
        if _matches(word, stems, preprocess):
            out.append(f"<mark>{escape(word)}</mark>")
        else:
            out.append(escape(word))
        pos = m.end()
    out.append(escape(text[pos:]))
    return "".join(out)


def best_snippet(abstract, stems, preprocess, max_sentences=2, max_chars=420):
    """Pick the sentences containing the most query terms (kept in original
    order). Falls back to the opening of the abstract when nothing matches."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(abstract) if s.strip()]
    if not sentences:
        return ""
    scored = []
    for i, s in enumerate(sentences):
        hits = sum(1 for w in _WORD.findall(s) if _matches(w, stems, preprocess))
        scored.append((hits, i))
    top = sorted(scored, key=lambda t: (-t[0], t[1]))[:max_sentences]
    if top[0][0] == 0:
        chosen = [0, 1][:max_sentences]
    else:
        chosen = sorted(i for hits, i in top if hits > 0)
    snippet = " ".join(sentences[i] for i in chosen if i < len(sentences))
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "…"
    elif chosen and chosen[0] > 0:
        snippet = "… " + snippet
    return snippet
