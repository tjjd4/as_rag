import argparse
import logging
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from get_embedding_function import get_embedding_function

logging.basicConfig(level=logging.WARNING)

CHROMA_PATH = "chroma"
model_name = "google/flan-t5-large"

prompt_template = PromptTemplate.from_template("""
    You are a helpful assistant. Answer the question concisely based only on relevant information in the provided context. Ignore any irrelevant sections or formatting.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_text", type=str, help="The initial query text.", default="")
    args = parser.parse_args()
    
    try:
        while True:
            # 如果有初始輸入，使用它，否則提示用戶輸入
            query_text = args.query_text or input("Please enter your query (or press Ctrl+C to exit): ").strip()
            
            if not query_text:
                logging.error("Query text cannot be empty. Please enter a valid query.")
            else:
                query_rag(query_text)
            
            # 清除初始參數值，讓後續輸入由用戶提供
            args.query_text = ""
    
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    results = db.similarity_search_with_score(query_text, k=3)
    if not results:
        logging.info("No relevant documents found.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content[:512] for doc, _score in results])
    prompt = prompt_template.format(context=context_text, question=query_text)
    logging.info("Prompt: ", prompt)

    response = generator(prompt, max_new_tokens=200, num_return_sequences=1)

    answer_text = response[0]["generated_text"]
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {answer_text}\nSources: {sources}"
    
    print(f"------\n{formatted_response}\n------\n")
    return answer_text

if __name__ == "__main__":
    main()
