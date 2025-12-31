import tempfile
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import build_rag_pipeline

load_dotenv()

st.set_page_config(
    page_title="Multilingual RAG Chatbot",
    layout="wide",
)

st.title("Multilingual RAG Chatbot for Agriculture Documents")
st.write(
    "Upload a PDF document to get started."
)

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = None

if "last_language" not in st.session_state:
    st.session_state.last_language = None

if st.button("🔄 Reset Conversation"):
    st.session_state.last_user_query = None
    st.session_state.last_language = None
    st.success("Conversation has been reset. You can start fresh.")

#File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

#Process PDF button
if uploaded_file and st.button("Process PDF"):
    with st.spinner("Processing PDF..."):
        #save uploaded pdf to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        #Build RAG pipeline
        st.session_state.rag_chain = build_rag_pipeline(
            pdf_path=pdf_path,
            persist_dir='chroma_db'
        )

        st.session_state.pdf_processed = True
        st.session_state.last_user_query = None  # reset conversation
    st.success("PDF processed successfully! You can now ask questions.")

#Question input
user_query=st.text_input(
    "Ask a question (any language):",
    placeholder="e.g. आलू की औसत उपज कितनी है?"
    )

#Answer Generation
if user_query:
    if not st.session_state.pdf_processed:
        st.warning("Please upload and process a PDF document first.")
    else:
        # with st.spinner("Generating answer..."):
        #     answer = st.session_state.rag_chain.invoke(user_query)

        # st.markdown("### Answer")
        # st.write(answer)
        final_query = user_query

        # If user gives a short follow-up like "आलू की"
        if (
            st.session_state.last_user_query
            and len(user_query.split()) <= 3
        ):
            final_query = (
                st.session_state.last_user_query + " " + user_query
            )

        with st.spinner("Generating answer..."):
            answer = st.session_state.rag_chain.invoke(final_query)

        st.markdown("### Answer")
        st.write(answer)

        # Store last query for next turn
        st.session_state.last_user_query = user_query