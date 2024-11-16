import os
import sys
import shutil
import glob
import gc

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from tart.TART.src.modeling_enc_t5 import EncT5ForSequenceClassification
from tart.TART.src.tokenization_enc_t5 import EncT5Tokenizer
import torch
import torch.nn.functional as F
import numpy as np

from generation import generate_with_loop

from sentence_transformers import SentenceTransformer

# from generation import generate_with_loop

database_path = "vectorDB_finetuned_6pdfs"

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

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=40)

    chunks = text_splitter.create_documents(texts)
    print(len(chunks))
    print(chunks[0])

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

def retrieve_with_re_ranker(user_query, num, model_path, model, tokenizer, query_no):
    embedding_model = MyEmbedding(model_path)
    
    chromadb = Chroma(embedding_function=embedding_model,
                      collection_name='coll_cosine',
                      collection_metadata={"hnsw:space": "cosine"},
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

    unique_results = list(unique_results)
    
    in_answer = "retrieve a passage that answers this question from some paper"

    final_result = unique_results[0]
    
    for i in range(1, len(unique_results)):
        features = tokenizer(['{0} [SEP] {1}'.format(in_answer, user_query), '{0} [SEP] {1}'.format(in_answer, user_query)], 
                             [final_result, unique_results[i]], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            scores = model(**features).logits
            normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]
        if np.argmax(normalized_scores) != 0:
            final_result = unique_results[i]

    result_dir = "results/9907/"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    
    result_file = "9907_tart_stella1.5B_100Q_1st_Rtv.txt"
    
    with open(result_dir+result_file, "a") as output_file:
        output_file.write("Result {} :".format(query_no))
        output_file.write("\n")
        output_file.write(final_result)
        output_file.write("\n")
        output_file.write("\n")
        output_file.close()
    
    return final_result

# run this python file only when a new vector DB is going to be set up
if __name__ == "__main__":
    query_dir = "questions/"
    query_file = "questions_100.txt"

    with open(query_dir + query_file, 'r') as fr:
        user_queries = fr.read().split("\n")
        
    embedding_model = 'finetune_embed/finetuned_stella1.5B_6pdfs'
    
    chunk_size = 200
    chunk_number = set_vector_db(chunk_size, embedding_model)
    
    num = 50
    
    # score = retrieve(user_query, num, embedding_model)
    
    # print()
    # print("Embedding Model = {} :".format(embedding_model))
    # print("average score = {}".format(score))
        
    retrieved_results = []

    model = EncT5ForSequenceClassification.from_pretrained("facebook/tart-full-flan-t5-xl")
    tokenizer =  EncT5Tokenizer.from_pretrained("facebook/tart-full-flan-t5-xl")

    model.eval()

    for i in range(len(user_queries)):
        query = user_queries[i]
        result = retrieve_with_re_ranker(query, num, embedding_model, model, tokenizer, i)
        retrieved_results.append(result)
        gc.collect()

    result_dir = "results/9907/"
    
    '''result_file = "tart_stella1.5B_100Q_1st_Rtv.txt"
    
    with open(result_dir+result_file, "r") as retrieved_file:
        retrieved_list = retrieved_file.read().split("Result ")
        for retrieved_result in retrieved_list:
            if "\n" not in retrieved_result:
                continue
            retrieved_results.append(retrieved_result.split("\n")[1])'''
    
    result_file = "9907_tart_stella1.5B_100Q_1st_Ans.txt"
    
    with open(result_dir+result_file, "w") as output_file:
        for i in range(len(retrieved_results)):
            histories = ""

            retrieved_result = retrieved_results[i]

            generation_reranker = generate_with_loop("Here is a question : " + user_queries[i] + " And I give you a related document : " + retrieved_result + " Generate a answer for me.", histories)

            answer_reranker = ""

            for ans in generation_reranker:
                answer_reranker = ans
            
            output_file.write("Answer {} :".format(i))
            output_file.write("\n")
            output_file.write(answer_reranker)
            output_file.write("\n")
            output_file.write("\n")
