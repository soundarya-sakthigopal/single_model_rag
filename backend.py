#from prompts import generate_prompt
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from IPython.display import display, Markdown
import os
# API Key is required
f = open("read from")
OPENAI_API_KEY = f.read()
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
client = OpenAI(api_key = OPENAI_API_KEY)

#Initialize embedding model and LLM model
openai_embed_model = OpenAIEmbeddings(model ="text-embedding-3-small")
openai_chat_model = ChatOpenAI(model_name="gpt-4o-mini")


loader = PyMuPDFLoader('./ragbook.pdf')
doc_pages = loader.load()

rag_prompt = """
You are an assistant specialized in question answering and translation.
Your task is to respond to the given question using only the information provided in the retrieved context.

Instructions:
- If the answer is not present in the context, clearly state: "I am sorry, but i don’t know based on the given context."
- Do not invent or assume any information.
- Write the answer in clear, simple language with correct grammar.
- Make the response detailed, structured, and easy to understand.

Question:
{question}

Context:
{context}

Answer:
"""

def generate_response(user_input):
    loader = PyMuPDFLoader('./ragbook.pdf')
    doc_pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    doc_chunks = splitter.split_documents(doc_pages)
    chroma_db = Chroma.from_documents(documents=doc_chunks,
                                  collection_name="my_db",
                                  embedding = openai_embed_model,
                                  collection_metadata = {"hnsw:space":"cosine"},
                                  persist_directory="/content/my_chromadb")

# LATER ON WE JUST ACCESS IT
    chroma_db = Chroma(persist_directory="/content/my_chromadb",
                    embedding_function=openai_embed_model,
                    collection_name="my_db")
    similarity_retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k":5})
    
    top_docs = similarity_retriever.invoke(user_input)
     # --------------------------------
    # Retrieve documents WITH scores
    # --------------------------------

    results_with_scores = chroma_db.similarity_search_with_score(
        user_input,
        k=5
    )

    print("\n===== Retrieved Documents =====")

    for i, (doc, score) in enumerate(results_with_scores, start=1):

        print(f"\nDocument {i}")
        print(f"Distance Score: {score}")
        print(f"Content: {doc.page_content[:300]}")

    rag_prompt_template = ChatPromptTemplate.from_template(rag_prompt)
    qa_rag_chain = (
    {"context":(similarity_retriever | format_docs),
     "question": RunnablePassthrough()} | rag_prompt_template | openai_chat_model
    )
    result = qa_rag_chain.invoke(user_input)
    return result.content

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

