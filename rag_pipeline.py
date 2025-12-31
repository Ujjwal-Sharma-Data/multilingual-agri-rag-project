import os
from typing import List

from llama_parse import LlamaParse

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_classic.schema import BaseRetriever


# ---------------------------------------------------------
# Helper: merge markdown from LlamaParse documents
# ---------------------------------------------------------
def full_markdown_text(docs):
    return "\n\n".join(doc.text for doc in docs)


# ---------------------------------------------------------
# Helper: clean retrieved context
# ---------------------------------------------------------
def clean_context_fn(docs: List[Document]) -> str:
    return "\n\n".join(
        doc.page_content.replace("passage:", "").strip()
        for doc in docs
    )


clean_context = RunnableLambda(clean_context_fn)


# ---------------------------------------------------------
# Custom retriever for BGE query prefix
# ---------------------------------------------------------
class BGEQueryPrefixRetriever(BaseRetriever):
    retriever: BaseRetriever

    def _get_relevant_documents(self, query: str) -> List[Document]:
        prefixed_query = "query: " + query
        return self.retriever.invoke(prefixed_query)


# ---------------------------------------------------------
# MAIN PIPELINE (ALL HEAVY OBJECTS CREATED INSIDE FUNCTION)
# ---------------------------------------------------------
def build_rag_pipeline(
    pdf_path: str,
    persist_dir: str = "chroma_db",
):
    # -------------------------------
    # 1. LlamaParse (PDF → Markdown)
    # -------------------------------
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        parsing_instruction=(
            "Extract all text, preserve tables accurately, "
            "keep headings and section structure."
        ),
    )

    docs = parser.load_data(pdf_path)

    # -------------------------------
    # 2. Merge markdown
    # -------------------------------
    markdown_text = full_markdown_text(docs)

    # -------------------------------
    # 3. Markdown header splitting
    # -------------------------------
    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    chunked_texts = markdown_splitter.split_text(markdown_text)

    # -------------------------------
    # 4. Prefix passages (BGE rule)
    # -------------------------------
    prefixed_docs = [
        Document(
            page_content="passage: " + chunk.page_content,
            metadata=chunk.metadata,
        )
        for chunk in chunked_texts
    ]

    # -------------------------------
    # 5. Embeddings (CPU ONLY for cloud)
    # -------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # -------------------------------
    # 6. Vector store (Chroma)
    # -------------------------------
    vectorstore = Chroma.from_documents(
        documents=prefixed_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    # -------------------------------
    # 7. Retriever (MMR)
    # -------------------------------
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.5},
    )

    final_retriever = BGEQueryPrefixRetriever(
        retriever=base_retriever
    )

    # -------------------------------
    # 8. LLM (Gemini)
    # -------------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    # -------------------------------
    # 9. Prompt
    # -------------------------------
    prompt = PromptTemplate(
        template="""
    You are an expert agricultural assistant.

    RULES:
    1. Use ONLY the provided context.
    2. You may derive answers from tables or structured data.
    3. Answer strictly in the same language as the user's question.
    4. If the question is ambiguous, ask a clarification question.
    5. If the context has no relevant information, say: "I don't know."

    Context:
    {context}
    -----------------------------------------------------------
    Question:
    {question}
    """,
        input_variables=["context", "question"],
    )

    # -------------------------------
    # 10. Runnable RAG chain
    # -------------------------------
    parallel_chain = RunnableParallel(
        {
            "context": final_retriever | clean_context,
            "question": RunnablePassthrough(),
        }
    )

    rag_chain = ( parallel_chain | prompt | llm | StrOutputParser() )

    return rag_chain
