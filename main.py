import argparse
import os
from src.rag_pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', type=str, required=True, help='Path to document folder')
    parser.add_argument('--query', type=str, required=True, help='Query string')
    args = parser.parse_args()

    rag = RAGPipeline()
    for file in os.listdir(args.docs):
        if file.endswith('.pdf'):
            rag.index_document(os.path.join(args.docs, file))

    answer = rag.answer_question(args.query)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
