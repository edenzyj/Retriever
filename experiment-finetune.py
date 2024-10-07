import os
import sys
import time
import shutil
import glob
from tika import parser
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

from tart.TART.src.modeling_enc_t5 import EncT5ForSequenceClassification
from tart.TART.src.tokenization_enc_t5 import EncT5Tokenizer
import torch
import torch.nn.functional as F
import numpy as np

from generation import generate_with_loop

from sentence_transformers import SentenceTransformer

database_path = "vectorDB_test"

class MyEmbedding:
    def __init__(self, model):
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, query: str) -> List[float]:
        encoded_query = self.model.encode(query)
        return self.model.encode(query).tolist()


def set_vector_db(chunk_size, model_path):
    pdf_dir = 'pdf/strawberry_file/EN'
    file_names = glob.glob(pdf_dir + "/*.pdf")
    
    texts = []
    
    for file_name in file_names:
        text = parser.from_file(file_name)
        pdf_str = text["content"]
        
        new_str = ""
        
        for i in range(len(pdf_str)):
            if pdf_str[i] == '-' and pdf_str[i+1] == '\n':
                continue
            if pdf_str[i] != '\n':
                new_str = new_str + pdf_str[i]
                continue
            if i == 0:
                continue
            if pdf_str[i-1] == '.':
                first_letter = str(new_str.split(' ')[-1][0])
                if not first_letter.isupper():
                    new_str = new_str + pdf_str[i]
                continue
        
        new_list = new_str.split('\n')
        
        for split_str in new_list:
            texts.append(split_str)

    '''# Use local finetuned model
    embeddings_model = SentenceTransformer(embedding_model, trust_remote_code=True)
    embedded_texts = embeddings_model.encode(texts)
    
    if os.path.isdir(database_path):
        shutil.rmtree(database_path)
        
    os.makedirs(database_path)

    client = chromadb.PersistentClient(path = database_path)

    collection = client.create_collection(name="finetune", metadata={"hnsw:space": "cosine"})

    collection.add(documents=texts,
                   embeddings=embedded_texts,
                   ids=["paragraph{}".format(i) for i in range(len(texts))])'''

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=40)

    chunks = text_splitter.create_documents(texts)

    embedding_model = MyEmbedding(model_path)

    if os.path.isdir(database_path):
        shutil.rmtree(database_path)
        
    os.makedirs(database_path)

    chromadb = Chroma.from_documents(chunks, 
                                     embedding=embedding_model,
                                     collection_name='coll_cosine',
                                     collection_metadata={"hnsw:space": "cosine"},
                                     persist_directory=database_path)
    chromadb.persist()
    
    return len(chunks)

def retrieve(user_query, num, embedding_model):
    print(user_query)
    print()
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name = embedding_model,
        model_kwargs = {'device': 'cuda'},
        encode_kwargs = {'normalize_embeddings': False}
    )
    
    chromadb = Chroma(embedding_function=embeddings_model,
                      collection_name='coll_cosine',
                      collection_metadata={"hnsw:space": "cosine"},
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

    total_score = 0

    for i in range(first_num):
        total_score = total_score + final_results[i][1]
    
    avrg_score = total_score / first_num
    
    return avrg_score

def retrieve_with_re_ranker(user_query, num, model_path):
    '''# Use local finetuned model
    embeddings_model = SentenceTransformer(embedding_model, trust_remote_code=True)
    
    client = chromadb.PersistentClient(path=database_path)

    collection = client.get_collection(name="finetune")

    results = collection.query(query_texts=[user_query], n_results=50)["documents"][0]'''

    # Use local finetuned model
    embedding_model = MyEmbedding(model_path)
    
    chromadb = Chroma(embedding_function=embedding_model,
                      collection_name='coll_cosine',
                      collection_metadata={"hnsw:space": "cosine"},
                      persist_directory=database_path)

    results = chromadb.similarity_search_with_score(user_query, num)
    
    model = EncT5ForSequenceClassification.from_pretrained("facebook/tart-full-flan-t5-xl")
    tokenizer =  EncT5Tokenizer.from_pretrained("facebook/tart-full-flan-t5-xl")

    model.eval()
    
    in_answer = "retrieve a passage that answers this question from some paper"

    final_results = []
    
    for result in results:
        final_results.append([result[0].page_content, 0])
    
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            features = tokenizer(['{0} [SEP] {1}'.format(in_answer, user_query), '{0} [SEP] {1}'.format(in_answer, user_query)], 
                                 [results[i][0].page_content, results[j][0].page_content], padding=True, truncation=True, return_tensors="pt")
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
        final_results = final_results[:10]

    result_dir = "results/"
    result_file = "tart_finetune1.5B_3.txt"
    
    if os.path.isfile(result_dir+result_file):
        os.remove(result_dir+result_file)
    
    with open(result_dir+result_file, "w") as output_file:
        for i in range(first_num):
            output_file.write("Result {} :".format(i))
            output_file.write("\n")
            output_file.write(final_results[i][0])
            output_file.write("\n")
            output_file.write("\n")
    
    return final_results

# run this python file only when a new vector DB is going to be set up
if __name__ == "__main__":
    user_query = "What are the most effective methods for preventing and controlling anthracnose in strawberry crops?"
    
    embedding_model = 'finetune_embed/epoch_5_20241007'
    
    chunk_size = 200
    chunk_number = set_vector_db(chunk_size, embedding_model)
    
    num = 50
    
    # score = retrieve(user_query, num, embedding_model)
    
    # print()
    # print("Embedding Model = {} :".format(embedding_model))
    # print("average score = {}".format(score))
        
    retrieved_results = retrieve_with_re_ranker(user_query, num, embedding_model)
    
    result_dir = "results/"
    result_file = "tart_finetune1.5B_generation_3.txt"
    
    with open(result_dir+result_file, "w") as output_file:
        for i in range(len(retrieved_results)):
            histories = ""

            retrieved_result = retrieved_results[i][0]

            generation_reranker = generate_with_loop(user_query + " " + retrieved_result, histories)

            answer_reranker = ""

            for ans in generation_reranker:
                answer_reranker = ans
            
            output_file.write("Answer {} :".format(i))
            output_file.write("\n")
            output_file.write(answer_reranker)
            output_file.write("\n")
            output_file.write("\n")
