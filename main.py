import argparse
from src.rag_pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="RAG-based PDF QA System using Qdrant")
    parser.add_argument('--docs', type=str, required=True, help='Path to folder containing PDF documents')
    parser.add_argument('--query', type=str, required=True, help='Query string to search in indexed documents')
    args = parser.parse_args()

    # Initialize RAG pipeline (will only index once if Qdrant is empty)
    rag = RAGPipeline()

    # Generate answer from query
    answer = rag.answer_question(args.query)
    print("\n📌 Answer:", answer)

if __name__ == "__main__":
    main()