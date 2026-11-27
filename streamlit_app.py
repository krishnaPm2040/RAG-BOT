"""
streamlit_app.py - web chat UI for the RBI RAG chatbot

  python -m streamlit run streamlit_app.py

Opens http://localhost:8501 in your browser.

- Every browser tab has its OWN settings (they are passed with each question,
  nothing global is changed), so two users don't change each other's settings.
- Follow-up questions work: "and for rural accounts?" uses the last answers.
- The UI only OPENS databases. Build them first with:  python app.py
"""


import streamlit as st

import app   # your app.py: vector DB, retrieval, generation with citations

st.set_page_config(page_title="Document Chatbot", page_icon="🏦", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: this tab's settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    # Only the databases that are built
    built = [n for n in ("recursive", "fixed", "semantic") if app.is_built(n)] or ["recursive"]
    strategy = st.selectbox("Chunking strategy", built,
                            help="Each strategy has its own database. Build them with: python app.py")
    scope = "all"              # annual reports are not in the database, so no "Search in" choice
    use_hybrid = st.checkbox("Hybrid search (BM25 + vectors)", value=True)
    use_rerank = st.checkbox("Re-ranker (keeps 5 sources)", value=False,
                             help="First use downloads a small model (~90 MB).")
    use_rewrite = st.checkbox("Query rewriting", value=False)
    show_chunks = st.checkbox("Show retrieved text under sources", value=False)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Vector database + re-ranker: opened once, shared (read-only) by all tabs
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_db(strategy_name):
    return app.get_vector_db(strategy=strategy_name)


@st.cache_resource(show_spinner="Loading the chat model (first time only)...")
def warm_up_model():
    app.warm_up()
    return True


@st.cache_resource(show_spinner=False)
def load_reranker():
    return app.get_reranker()


st.title(f"🏦 {app.DOMAIN} chatbot")
st.caption("Answers come only from your documents, with the file and page for every source.")

if not app.is_built(strategy):                       # the UI never builds: app.py does that
    st.warning(f"The '{strategy}' database is not built yet (or the documents changed).\n\n"
               f"In the terminal run:  `python app.py --strategy={strategy}`  then reload this page.")
    st.stop()

try:
    with st.spinner(f"Opening the '{strategy}' vector database..."):
        vector_db = load_db(strategy)
    warm_up_model()                       # loads qwen into memory once, before the first question
    if use_rerank:
        with st.spinner("Loading the re-ranker..."):
            load_reranker()
except SystemExit as e:                   # app.py stops with sys.exit() if output/json is empty
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not open the vector database: {e}\n\n"
             + "Is Ollama running? Did you run `ollama pull nomic-embed-text`?")
    st.stop()

with st.sidebar:
    st.divider()
    try:
        st.caption(f"Chunks in database: {vector_db._collection.count():,}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def safe(text):
    return text.replace("$", "\\$")       # "$...$" must not turn into maths


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=True):
        for s in sources:
            line = f"**[{s['n']}]** {s['source']} — page {s['page']}"
            if s.get("circular_no"):
                line += f" | {s['circular_no']}"
            if s.get("issue_date"):
                line += f" ({s['issue_date']})"
            st.markdown(safe(line))
            if s.get("heading"):
                st.caption(s["heading"])
            if show_chunks:
                st.text(s["text"][:1500])


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:                 # earlier messages
    with st.chat_message(msg["role"]):
        st.markdown(safe(msg["content"]))
        if msg.get("understood_as"):
            st.caption(f"Understood as: {msg['understood_as']}")
        show_sources(msg.get("sources"))

question = st.chat_input("Ask a question about your documents...")
if question:
    # last turns of THIS tab, for follow-up questions
    history = [(m["question"], m["content"]) for m in st.session_state.messages
               if m["role"] == "assistant" and m.get("question")]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(safe(question))

    with st.chat_message("assistant"):
        live = st.empty()                                  # the answer appears here word by word
        with st.spinner("Searching the documents..."):
            try:
                reply = app.answer_question(vector_db, question, history,
                                            hybrid=use_hybrid, rerank_on=use_rerank,
                                            rewrite=use_rewrite,
                                            scope=scope,
                                            on_token=lambda text: live.markdown(safe(text) + " ▌"))
            except Exception as e:
                reply = {"answer": f"Error: {e}", "results": [], "cited": set(),
                         "invalid_citations": [], "question_used": question}

        sources = [{"n": i,
                    "source": d.metadata.get("source", ""),
                    "page": d.metadata.get("page", ""),
                    "circular_no": d.metadata.get("circular_no", ""),
                    "issue_date": d.metadata.get("issue_date", ""),
                    "heading": d.metadata.get("heading", ""),
                    "text": d.page_content}
                   for i, d in app.sources_to_show(reply["answer"], reply["results"])]
                                                    # cited sources only; none for a refusal (L2)
        understood = reply["question_used"] if reply["question_used"] != question else ""

        live.markdown(safe(reply["answer"]))
        if understood:
            st.caption(f"Understood as: {understood}")
        if reply.get("used_old_versions"):
            st.warning("No current document answers this. The answer comes from a WITHDRAWN or "
                       "superseded circular: check that it still applies.")
        if reply["invalid_citations"]:
            st.warning(f"The answer cites {reply['invalid_citations']}, "
                       f"but only {len(reply['results'])} sources exist.")
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": reply["answer"],
                                      "question": question, "understood_as": understood,
                                      "sources": sources})
