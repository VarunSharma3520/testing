from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


file_path = "./data/F18-ABCD-000.pdf"
loader = PyPDFLoader(file_path)
minerLoader = PDFMinerLoader(file_path)

minerDocs = minerLoader.load()
print(len(minerDocs))

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0,
)

response = llm.invoke("Hello, world!")
print(response.content)

doc_splits = text_splitter.split_documents(minerDocs)




def main():
    print("Hello from testrag!")


if __name__ == "__main__":
    main()
