from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedding_function():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    return embeddings