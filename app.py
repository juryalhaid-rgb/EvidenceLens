"""
EvidenceLens - web interface (Streamlit).

Run:  streamlit run app.py
Needs the PyTerrier index in the folder  ./index  (built by the notebook).
Every result shown comes from the real index through ir_core.search();
nothing is mocked.
"""
import glob
import os
from html import escape

import streamlit as st

import ir_core
from ui_helpers import best_snippet, highlight, query_stems

INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index")

# On Linux hosts (e.g. Streamlit Community Cloud) point PyTerrier at the Java
# that apt installed. On Windows/Anaconda JAVA_HOME is already set, so this is skipped.
if not os.environ.get("JAVA_HOME"):
    _libs = sorted(glob.glob("/usr/lib/jvm/*/lib/server/libjvm.so"))
    if _libs:
        os.environ["JVM_PATH"] = _libs[0]
        os.environ["JAVA_HOME"] = _libs[0].split("/lib/server")[0]
MODELS = ["BM25", "TF-IDF"]
EXAMPLES = [
    "Does regular exercise improve sleep quality?",
    "Is vitamin D deficiency associated with fatigue?",
    "What does scientific research say about intermittent fasting?",
]

st.set_page_config(page_title="EvidenceLens", page_icon="🔎", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
:root{--ink:#16222E;--muted:#5B6B79;--line:#D5DEE4;--teal:#0E7C86;--teal-soft:#D7EEF0;--card:#FFFFFF;}
html, body, [class*="css"], .stApp{font-family:'IBM Plex Sans',system-ui,sans-serif;color:var(--ink);}
.block-container{max-width:760px;padding-top:2.5rem;}
h1.el-title{font-family:'Newsreader',Georgia,serif;font-weight:600;font-size:2.6rem;letter-spacing:-0.01em;margin:0 0 .25rem 0;}
p.el-sub{color:var(--muted);font-size:1.02rem;line-height:1.5;margin:0 0 1.4rem 0;max-width:60ch;}
.el-note{border-left:3px solid var(--teal);padding:.15rem 0 .15rem .8rem;color:var(--muted);font-size:.88rem;margin:.9rem 0 1.4rem 0;}
.el-processed{color:var(--muted);font-size:.86rem;margin:.2rem 0 1rem 0;}
.el-processed code{background:var(--teal-soft);color:var(--ink);border-radius:4px;padding:1px 6px;}
.el-card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1.05rem 1.2rem 1rem 1.2rem;margin-bottom:.9rem;}
.el-head{display:flex;gap:.9rem;align-items:baseline;}
.el-rank{font-family:'Newsreader',Georgia,serif;font-size:1.7rem;font-weight:600;color:var(--teal);min-width:2rem;}
.el-card h3{font-family:'Newsreader',Georgia,serif;font-size:1.22rem;line-height:1.3;font-weight:600;margin:0;}
.el-card h3 a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--line);}
.el-card h3 a:hover{border-bottom-color:var(--teal);}
.el-meta{color:var(--muted);font-size:.84rem;margin:.35rem 0 .6rem 2.9rem;}
.el-scorewrap{display:flex;align-items:center;gap:.7rem;margin:0 0 .7rem 2.9rem;}
.el-bar{flex:0 0 150px;height:6px;background:var(--line);border-radius:3px;overflow:hidden;}
.el-bar > span{display:block;height:100%;background:var(--teal);}
.el-score{font-size:.82rem;color:var(--ink);font-weight:500;}
.el-snippet{margin:0 0 .5rem 2.9rem;font-size:.95rem;line-height:1.6;}
.el-card mark{background:var(--teal-soft);color:var(--ink);padding:0 2px;border-radius:3px;}
.el-card details{margin:.2rem 0 .3rem 2.9rem;font-size:.9rem;line-height:1.6;}
.el-card summary{cursor:pointer;color:var(--teal);font-weight:500;}
.el-link{margin:.4rem 0 0 2.9rem;font-size:.88rem;}
.el-link a{color:var(--teal);font-weight:500;}
@media (max-width:600px){.el-meta,.el-scorewrap,.el-snippet,.el-card details,.el-link{margin-left:0;}.el-bar{flex-basis:90px;}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the search index…")
def load_system():
    """Load the PyTerrier index and both retrievers once per session."""
    index = ir_core.load_index(INDEX_DIR)
    num_docs = index.getCollectionStatistics().getNumberOfDocuments()
    return num_docs, ir_core.get_retrievers(index)


def set_example(text):
    st.session_state["q"] = text
    st.session_state["run"] = True


def request_search():
    st.session_state["run"] = True


def render_result(row, stems, top_score):
    title = escape(str(row.title))
    url = escape(str(row.url), quote=True)
    pct = 0 if top_score <= 0 else max(4, round(100 * float(row.score) / top_score))
    snippet = highlight(best_snippet(str(row.abstract), stems, ir_core.preprocess),
                        stems, ir_core.preprocess)
    full = highlight(str(row.abstract), stems, ir_core.preprocess)
    meta = ", ".join(x for x in [escape(str(row.journal)), escape(str(row.year))] if x)
    st.markdown(f"""
<div class="el-card">
  <div class="el-head"><span class="el-rank">{int(row.rank) + 1}</span>
    <h3><a href="{url}" target="_blank" rel="noopener">{title}</a></h3></div>
  <div class="el-meta">{meta}, PubMed ID {escape(str(row.docno))}</div>
  <div class="el-scorewrap"><div class="el-bar"><span style="width:{pct}%"></span></div>
    <span class="el-score">Score {float(row.score):.2f}</span></div>
  <p class="el-snippet">{snippet}</p>
  <details><summary>Read the full abstract</summary><p>{full}</p></details>
  <div class="el-link"><a href="{url}" target="_blank" rel="noopener">Open on PubMed</a></div>
</div>""", unsafe_allow_html=True)


# ------------------------------- page ------------------------------------
if not os.path.isdir(INDEX_DIR):
    st.error("The search index was not found. Unzip index.zip so that the folder "
             "'index' sits next to app.py, then reload this page.")
    st.stop()

num_docs, retrievers = load_system()

st.markdown('<h1 class="el-title">EvidenceLens</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="el-sub">Search {num_docs:,} PubMed research abstracts to find '
            'scientific studies related to a health question.</p>', unsafe_allow_html=True)

with st.form("search_form"):
    st.text_input("Health question or topic", key="q",
                  placeholder="For example: Does regular exercise improve sleep quality?")
    st.form_submit_button("Search", type="primary", on_click=request_search)

# The model choice sits OUTSIDE the form so that switching it re-runs the
# search immediately (widgets inside a form only update when Search is pressed).
model = st.radio("Ranking model", MODELS, horizontal=True, key="model",
                 help="Both models rank the same index. BM25 adjusts for document "
                      "length; TF-IDF is the classic baseline.")

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, text in zip(cols, EXAMPLES):
    col.button(text, key=f"ex_{text}", on_click=set_example, args=(text,),
               use_container_width=True)

query = st.session_state.get("q", "").strip()
if st.session_state.get("run") and query:
    processed = ir_core.preprocess(query)
    if not processed.strip():
        st.warning("Your question only contains very common words. Add a more specific "
                   "health term, such as a condition, treatment or symptom.")
    else:
        results = ir_core.search(retrievers[model], query)
        st.markdown(f'<div class="el-processed">Searched {model} for the terms '
                    f'<code>{escape(processed)}</code> after removing common words and '
                    'reducing words to their stems.</div>', unsafe_allow_html=True)
        if results.empty:
            st.info("No studies matched these terms. Try different or fewer words.")
        else:
            stems = query_stems(query, ir_core.preprocess)
            top_score = float(results.score.max())
            for row in results.itertuples(index=False):
                render_result(row, stems, top_score)
            st.markdown('<div class="el-note">The score shows how closely a study matches '
                        'your words. It does not show that a health claim is true. EvidenceLens '
                        'helps you find research; it gives no medical advice or diagnosis. '
                        'Bars compare results within this list only.</div>',
                        unsafe_allow_html=True)
elif st.session_state.get("run"):
    st.info("Type a health question above, then press Search.")
else:
    st.markdown('<div class="el-note">EvidenceLens finds relevant research. It does not '
                'diagnose conditions or check whether health claims are true.</div>',
                unsafe_allow_html=True)
