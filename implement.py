import os
import shutil
import glob
from tika import parser

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

database_path = "vectorDB"

def set_vector_db():
    pdf_dir = 'pdf/starwberry_file/EN'
    file_names = glob.glob(pdf_dir + "/*.pdf")
    
    texts = []
    
    for file_name in file_names:
        text = parser.from_file(file_name)
        print(type(text["content"]))
        texts.append(text["content"])

    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=40)

    chunks = text_splitter.create_documents(texts)
    print(len(chunks))
    print(chunks[0])

    embeddings_model = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs = {'device': 'cuda'},
        encode_kwargs = {'normalize_embeddings': False}
    )
    
    if os.path.isdir(database_path):
        shutil.rmtree(database_path)

    chromadb = Chroma.from_documents(chunks, embeddings_model, persist_directory=database_path)
    chromadb.persist()
    
    return

def retrieve(user_query):
    print(user_query)
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs = {'device': 'cuda'},
        encode_kwargs = {'normalize_embeddings': False}
    )
    
    chromadb = Chroma(embedding_function=embeddings_model, persist_directory=database_path)

    results = chromadb.similarity_search_with_score(user_query, 50)
    
    unique_results = set()
    final_results = []

    for i in range(len(results)):
        content = results[i][0].page_content
        if content not in unique_results:
            unique_results.add(content)
            final_results.append((content, results[i][1]))
    
    final_results.sort(key=lambda a: a[1])

    return_message = user_query
    
    print("number of unique results : {}".format(len(unique_results)))
    print("=======================")

    '''for i in range(min(50, len(final_results))):
        print("Result {}, score = {} :".format(i, final_results[i][1]))
        print(final_results[i][0])
        print("=======================")'''
    
    if final_results[0][1] <= 0.8:
        print(final_results[0][1])
        print(len(final_results[0][0]))
        return_message = return_message + " " + final_results[0][0]
    
    # print(type(return_message))
    # print(return_message)
    
    return return_message

# run this python file only when a new vector DB is going to be set up
if __name__ == "__main__":
    set_vector_db()
    
    '''user_query = "What is Anthracnose caused by?"
    answer = retrieve(user_query)
    print(answer)'''
