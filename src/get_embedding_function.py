from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def get_embedding_function():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",  # alternatively use "sentence-transformers/all-MiniLM-l6-v2" for a light and faster experience.
        model_kwargs={'device':'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

