import os
import shutil
import glob
from tika import parser

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from tart.TART.src.modeling_enc_t5 import EncT5ForSequenceClassification
from tart.TART.src.tokenization_enc_t5 import EncT5Tokenizer
import torch
import torch.nn.functional as F
import numpy as np

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
        
    os.makedirs(database_path)

    chromadb = Chroma.from_documents(documents=chunks,
                                     embedding=embeddings_model,
                                     collection_name='coll_l2',
                                     collection_metadata={"hnsw:space": "l2"},
                                     persist_directory=database_path)
    chromadb.persist()
    
    return

def retrieve(user_query, num):
    print(user_query)
    print()
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs = {'device': 'cuda'},
        encode_kwargs = {'normalize_embeddings': False}
    )
    
    chromadb = Chroma(embedding_function=embeddings_model,
                      collection_name='coll_l2',
                      collection_metadata={"hnsw:space": "l2"},
                      persist_directory=database_path)

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

    result_dir = "results/"
    result_file = "l2_result.txt"
    
    if os.path.isfile(result_dir+result_file):
        os.remove(result_dir+result_file)
    
    with open(result_dir+result_file, "w") as output_file:
        for i in range(first_num):
            output_file.write("Result {}, with score = {} :".format(i, final_results[i][1]))
            output_file.write("\n")
            output_file.write(final_results[i][0])
            output_file.write("\n")
            output_file.write("\n")
        
    return

def retrieve_with_re_ranker(user_query, num):
    embeddings_model = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs = {'device': 'cuda'},
        encode_kwargs = {'normalize_embeddings': False}
    )
    
    chromadb = Chroma(embedding_function=embeddings_model,
                      collection_name='coll_l2',
                      collection_metadata={"hnsw:space": "l2"},
                      persist_directory=database_path)

    results = chromadb.similarity_search_with_score(user_query, num)
    
    unique_results = set()

    for i in range(len(results)):
        content = results[i][0].page_content
        if content not in unique_results:
            unique_results.add(content)
    
    print("number of unique results : {}".format(len(unique_results)))
    print("=======================")
    print()
    
    model = EncT5ForSequenceClassification.from_pretrained("facebook/tart-full-flan-t5-xl")
    tokenizer =  EncT5Tokenizer.from_pretrained("facebook/tart-full-flan-t5-xl")

    model.eval()
    
    in_answer = "retrieve a passage that answers this question from some paper"

    final_results = []
    
    for result in unique_results:
        final_results.append([result, 0])
    
    unique_results = list(unique_results)
    for i in range(len(unique_results)):
        for j in range(i+1, len(unique_results)):
            features = tokenizer(['{0} [SEP] {1}'.format(in_answer, user_query), '{0} [SEP] {1}'.format(in_answer, user_query)], 
                                 [unique_results[i], unique_results[j]], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                scores = model(**features).logits
                normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]
            if np.argmax(normalized_scores) == 0:
                final_results[i][1] = final_results[i][1] + 1
            else:
                final_results[j][1] = final_results[j][1] + 1
    
    final_results.sort(reverse=True, key=lambda a: a[1])
    
    if len(final_results) < 10:
        first_num = len(final_results)
    else:
        first_num = 10

    result_dir = "results/"
    result_file = "re-ranker_result.txt"
    
    if os.path.isfile(result_dir+result_file):
        os.remove(result_dir+result_file)
    
    with open(result_dir+result_file, "w") as output_file:
        for i in range(first_num):
            output_file.write("Result {} :".format(i))
            output_file.write("\n")
            output_file.write(final_results[i][0])
            output_file.write("\n")
            output_file.write("\n")
        
    return

# run this python file only when a new vector DB is going to be set up
if __name__ == "__main__":
    # set_vector_db()
    
    user_query = "What is Anthracnose caused by?"
    
    num = 50
    # retrieve(user_query, num)
    
    retrieve_with_re_ranker(user_query, num)
