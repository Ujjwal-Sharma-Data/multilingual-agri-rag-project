import os
from llama_parse import LlamaParse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema import BaseRetriever
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

#Load Models
parser=LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown", #Required for tables
    parsing_instruction=(
        "Extract all text, preserve tables accurately, "
        "keep headings and section structure."
    )
)

llm=ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={
        "device": "mps" #for MAC
    },
    encode_kwargs={
        "normalize_embeddings": True
        }
    )

prompt=PromptTemplate(
    template="""
You are an expert agricultural assistant.
RULES:
1. Use only the provided context to answer.
2. You may derive answers from tables or structured data in the context.
3. Answer strictly in the same language as the user's question.
4. If the question is ambiguous or could have multiple valid answers,
   ask a clarification question before answering.
5. Say "I don't know" ONLY if the context has no relevant information at all.


Context: {context}
-----------------------------------------------------------
Question:{question}
""",
input_variables=["context", "question"]
)

#Helpers
def full_markdown_text(docs):
    markdown_text="\n\n".join([doc.text for doc in docs])
    return markdown_text

from langchain_text_splitters import MarkdownHeaderTextSplitter
headers_to_split_on=[
    ('#', 'H1'),
    ('##', 'H2'),
    ('###', 'H3'),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)


class BGEQueryPrefixRetriever(BaseRetriever):
    retriever: BaseRetriever #This declares the retriever
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Modify the query to include the BGE prefix
        prefixed_query="query: "+query
        return self.retriever.invoke(prefixed_query)

def clean_context_fn(docs):
    return "\n\n".join(doc.page_content.replace("passage:", "").strip() for doc in docs)

clean_context = RunnableLambda(clean_context_fn)

#full pipeline
def build_rag_pipeline(
    pdf_path: str,
    persist_dir: str = "chroma_db",
):
    #Load PDF
    docs = parser.load_data(pdf_path)

    #Merge markdown (your function)
    markdown_text = full_markdown_text(docs)

    #Split using headers (your splitter)
    chunked_text = markdown_splitter.split_text(markdown_text)

    #Add passage prefix (BGE requirement)
    prefixed_docs = [
        Document(
            page_content="passage: " + chunk.page_content,
            metadata=chunk.metadata,
        )
        for chunk in chunked_text
    ]

    #Build vectorstore
    vectorstore = Chroma.from_documents(
        documents=prefixed_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    #Base retriever (MMR)
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.5},
    )

    #Prefix retriever
    final_retriever = BGEQueryPrefixRetriever(retriever=base_retriever)

    #Runnable chain
    parallel_chain = RunnableParallel(
        {
            "context": final_retriever | clean_context,
            "question": RunnablePassthrough(),
        }
    )

    rag_chain = parallel_chain | prompt | llm | StrOutputParser()

    return rag_chain

