import os
import argparse
from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


# --------------------------------------------------------
# Step 1: Build or load the Chroma vector store
# --------------------------------------------------------
def prepare_vectorstore(speech_path, db_path):
    # Load raw text
    loader = TextLoader(speech_path)
    docs = loader.load()

    # Split text into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    # Create HF embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create or load Chroma DB
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    vectordb.persist()
    return vectordb


# --------------------------------------------------------
# Step 2: Build the RetrievalQA chain
# --------------------------------------------------------
def build_chain(vectordb):
    llm = Ollama(model="mistral")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
    )
    return chain


# --------------------------------------------------------
# Step 3: Simple CLI loop
# --------------------------------------------------------
def run_cli(chain):
    print("\nAmbedkarGPT — Ask anything about the speech. Type 'exit' to stop.\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            out = chain(question)
        except Exception as e:
            print("Error:", e)
            continue

        print("\nAnswer:\n", out.get("result", "(no output)"))

        # Print supporting chunks
        docs = out.get("source_documents", [])
        if docs:
            print("\n--- Context Used ---")
            for i, d in enumerate(docs, 1):
                text = d.page_content.replace("\n", " ")
                preview = text[:250] + ("..." if len(text) > 250 else "")
                print(f"[{i}] {preview}")
            print()


# --------------------------------------------------------
# Entry point
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech', default='speech.txt')
    parser.add_argument('--db', default='chroma_db')
    parser.add_argument('--rebuild', action='store_true')

    args = parser.parse_args()

    # Rebuild the vector DB if requested
    if args.rebuild and os.path.isdir(args.db):
        import shutil
        shutil.rmtree(args.db)
        print("Rebuilding vector store...")

    vectordb = prepare_vectorstore(args.speech, args.db)
    chain = build_chain(vectordb)
    run_cli(chain)


if __name__ == '__main__':
    main()
