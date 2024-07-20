import os
import time
import shutil
import glob
from tika import parser

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

database_path = "vectorDB_test"

def set_vector_db(chunk_size):
    pdf_dir = 'pdf/starwberry_file/EN'
    file_names = glob.glob(pdf_dir + "/*.pdf")
    
    texts = []
    
    for file_name in file_names:
        text = parser.from_file(file_name)
        print(type(text["content"]))
        texts.append(text["content"])

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=40)

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
        
    os.makedirs(database_path)

    chromadb = Chroma.from_documents(chunks, embeddings_model, persist_directory=database_path)
    chromadb.persist()
    
    return len(chunks)

def retrieve(user_query, num):
    print(user_query)
    print()
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs = {'device': 'cuda'},
        encode_kwargs = {'normalize_embeddings': False}
    )
    
    chromadb = Chroma(embedding_function=embeddings_model, persist_directory=database_path)

    results = chromadb.similarity_search_with_score(user_query, num)
    
    unique_results = set()
    final_results = []

    for i in range(len(results)):
        content = results[i][0].page_content
        if content not in unique_results:
            unique_results.add(content)
            final_results.append((content, results[i][1]))
    
    final_results.sort(key=lambda a: a[1])
    
    print("number of unique results : {}".format(len(unique_results)))
    print("=======================")
    print()
    
    if len(final_results) < 10:
        first_num = len(final_results)
    else:
        first_num = 10

    total_score = 0

    for i in range(first_num):
        total_score = total_score + final_results[i][1]
    
    avrg_score = total_score / first_num
    
    return avrg_score

# run this python file only when a new vector DB is going to be set up
if __name__ == "__main__":
    user_query = "What is Anthracnose caused by?"
    
    chunk_size = 1000
    chunk_number = set_vector_db(chunk_size)
    
    num = 10
    score = retrieve(user_query, num)
    
    print()
    print("Chunk size = {} :".format(chunk_size))
    print("Number of chunks = {}".format(chunk_number))
    print("average score = {}".format(score))
