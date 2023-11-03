import streamlit as st

#Embedding
############# 이것도 함수로 따로 wrap하는게 나을듯?
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embedding = SentenceTransformerEmbeddings(model_name='BM-K/KoSimCSE-roberta', model_kwargs={'device':'cuda'}, encode_kwargs={'normalize_embeddings':True})


#Chroma Settings
#이건 따로 script 빼서 하는게 관리에 좋을듯? 아님말고
#persistentClient 관리하려면 지우는거떄문에 둘이 묶는게 나을수도
import chromadb
from langchain.vectorstores import Chroma

#VectorDB settings
chroma_client = chromadb.PersistentClient()

collection_name = "vector_db"
collection = chroma_client.get_collection(collection_name)

vectorstore = Chroma(
    client= chroma_client,
    collection_name= collection_name,
    embedding_function= embedding,
    persist_directory="../chroma"
)

#### Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

from langchain.retrievers import ParentDocumentRetriever

child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore= vectorstore,
    docstore=store,
    child_splitter= child_splitter,
    parent_splitter= parent_splitter,
)

def run_search(query) : 
    query = embedding.embed_query(query)
    results = vectorstore.max_marginal_relevance_search_by_vector(query,k=10,fetch_k=10)
    return results

if __name__ == "__main__" :
    st.title('SSiS Project Demo')
    search_query = st.text_input("검색어를 입력하세요:")
    search_button = st.button("검색")
    
    if search_query and search_button :
        
        # print("There are", vectorstore._collection.count(), "in the collection.")
        results = run_search(search_query)

        for result in results :
            st.title(f'**{result.metadata["title"]}**')
            st.markdown(result.page_content)
            st.write(result.metadata['tag'].split(','))

    # if search_button :
    #     print(search_query)
    #     print(run_search("search_query"))
