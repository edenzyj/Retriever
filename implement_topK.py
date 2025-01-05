import os
import shutil
import glob
import gc
from typing import List
import json

# Import langchain frameware to build vector DB of RAG.
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Import transformer to load finetuned embedding model from loacl.
from transformers import AutoModel, AutoTokenizer

# Import tart to implement reranker.
# Torch and Numpy is needed when using tart to rerank.
from tart.TART.src.modeling_enc_t5 import EncT5ForSequenceClassification
from tart.TART.src.tokenization_enc_t5 import EncT5Tokenizer
import torch
import torch.nn.functional as F
import numpy as np

# Import function to generate answer using llama3.
from generation import generate_with_loop

# Import configuration file
import config


class MyEmbedding:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze(0).tolist())
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).tolist()
        return embedding


def set_vector_db(file_names, chunk_size, use_finetuned, embedding_model, database_path):
    """
    Use vector store with embedding model to build vector DB.

    Parameters:
        file_names (list[str]): A list including the file names of all revised paragraphs' text files. 
        chunk_size (int): A number representing the approximate length of every chunk.
        use_finetuned (bool): A signal representing use finetuned embedding model or not.
        embedding_model (str): The repo name of the embedding model on HuggingFace or the directory path of the embedding model if the model is stored locally.
        database_path (str): The directory name of vector database.

    Returns:
        int: The number of chunks.
    """
    
    texts = []
    
    for file_name in file_names:
        with open(file_name, 'r') as fr:
            content = fr.read()
            paragraphs = content.split("\n")

            for paragraph in paragraphs:
                if len(paragraph) > 0:
                    texts.append(paragraph)

            fr.close()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=40)

    chunks = text_splitter.create_documents(texts)
    print(len(chunks))
    print(chunks[0])

    if use_finetuned:
        embeddings_model = MyEmbedding(embedding_model)
    else:
        embeddings_model = HuggingFaceEmbeddings(
            model_name = embedding_model,
            model_kwargs = {'device': 'cuda'},
            encode_kwargs = {'normalize_embeddings': False}
        )
    
    if os.path.isdir(database_path):
        shutil.rmtree(database_path)
        
    os.makedirs(database_path)

    chromadb = Chroma.from_documents(chunks, 
                                     embedding=embeddings_model,
                                     collection_name='coll_cosine',
                                     collection_metadata={"hnsw:space": "cosine"},
                                     persist_directory=database_path)
    chromadb.persist()
    
    return len(chunks)


def retrieve(user_query, num, use_finetuned, embedding_model, k):
    """
    Retrieve the results from vector DB using smilarity search with score, and then compare the scores to select the best retrieved result.

    Parameters:
        user_query (str): The query given by user.
        num (int): The number of results get from similarity search.
        use_finetuned (bool): A signal representing use finetuned embedding model or not.
        embedding_model (str): The repo name of the embedding model on HuggingFace or the directory path of the embedding model if the model is stored locally.
        k (int): The number of retrieved results merged.

    Returns:
        list[str]: The top k retrieved results which are ranked by score of similarity search.
    """
    
    if use_finetuned:
        embeddings_model = MyEmbedding(embedding_model)
    else:
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
    
    retrieved_results = [res[0] for res in final_results]
    
    return retrieved_results[:k]


def retrieve_with_re_ranker(user_query, num, use_finetuned, embedding_model, reranker_model, reranker_tokenizer, k):
    """
    Retrieve the results from vector DB using smilarity search, and then use reranker to select the best retrieved result.

    Parameters:
        user_query (str): The query given by user.
        num (int): The number of results get from similarity search.
        use_finetuned (bool): A signal representing use finetuned embedding model or not.
        embedding_model (str): The repo name of the embedding model on HuggingFace or the directory path of the embedding model if the model is stored locally.
        reranker_model : The reranker model loaded outside the function.
        reranker_tokenizer : The reranker tokenizer loaded outside the function.
        k (int): The number of retrieved results merged.
        
    Returns:
        list[str]: The top k retrieved results which are ranked by reranker.
    """
    
    if use_finetuned:
        embeddings_model = MyEmbedding(embedding_model)
    else:
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

    for i in range(len(results)):
        content = results[i][0].page_content
        if content not in unique_results:
            unique_results.add(content)
    
    print("number of unique results : {}".format(len(unique_results)))
    print("=======================")
    print()

    unique_results = list(unique_results)
    
    in_answer = "retrieve a passage that answers this question from some paper"
    
    features = reranker_tokenizer(
        ['{0} [SEP] {1}'.format(in_answer, user_query)] * len(unique_results),
        unique_results,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=4096
    )
    
    with torch.no_grad():
        scores = reranker_model(**features).logits
        normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]
        
    final_results = zip(normalized_scores, unique_results)
    final_results.sort(reverse=True, key=lambda a: a[0])
    
    retrieved_results = [res for score, res in final_results]
    
    return retrieved_results[:k]


# run this python file only when a new vector DB is going to be set up
if __name__ == "__main__":
    # =====Setting Here=====
    # Set the path of vector DB.
    database_path = config.database_path
    
    # =====Setting Here=====
    # These are parameters used to build vector DB.
    paragraph_dir = config.paragraph_directory
    file_names = glob.glob(paragraph_dir + "/*.txt")
    chunk_size = config.chunk_size
    embedding_model = config.embedding_model_path
    use_finetuned = config.use_finetuned_model
    
    # chunk_number = set_vector_db(file_names, chunk_size, use_finetuned, embedding_model, database_path)
    
    # print("Number of chunks : ".format(chunk_number))
    
    # =====Setting Here=====
    # Directory name and file name of query file.
    query_dir = config.query_directory
    query_file = config.query_file

    with open(query_dir + query_file, 'r') as fr:
        user_queries = fr.read().split("\n")
    
    retrieved_results = []  # List to store all retrieved results.
    num = 50  # Number of similarity search results.

    # Reranker model : TART from Facebook.
    if config.reranker is not None:
        reranker_tokenizer =  EncT5Tokenizer.from_pretrained(config.reranker)
        reranker_model = EncT5ForSequenceClassification.from_pretrained(config.reranker)
        reranker_model.eval()

    # =====Setting Here=====
    # Directory name of both retrieved results file and generated answers file.
    output_dir = config.output_directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    # =====Setting Here=====
    # File name of retrieved results file.
    result_file = config.result_file
    
    json_results = []
    
    # =====Setting Here=====
    # The number of retrieved results merged.
    top_k = config.top_k

    # Retrieve document and get result for each query.
    for i in range(len(user_queries)):
        query = user_queries[i]
        if config.reranker is not None:
            # Retrieve with reranker
            results = retrieve_with_re_ranker(query, num, use_finetuned, embedding_model, reranker_model, reranker_tokenizer, top_k)
        else:
            # Naive retrieve
            results = retrieve(query, num, use_finetuned, embedding_model)
        
        retrieved_results.append(results)
        
        json_results.append({
            "qid": i,
            "query": query,
            "retrieved_context": results
        })
        
        gc.collect()
    
    with open(output_dir + result_file, 'w') as output_file:
        json.dump(json_results, output_file, indent=4)
        output_file.close()
    
    '''
    # Seperate retrieving and generating.
    # Read result file to fulfill "retrieved_results" list.
    with open(output_dir+result_file, "r") as retrieved_file:
        retrieved_list = retrieved_file.read().split("Result ")
        for retrieved_result in retrieved_list:
            if "\n" not in retrieved_result:
                continue
            retrieved_results.append(retrieved_result.split("\n")[1])
    '''
    
    # =====Setting Here=====
    # File name of generated answers file.
    answer_file = config.answer_file
    
    # Generate answer and write into answer file for each retrieved result.
    with open(output_dir + answer_file, "w") as output_file:
        for i in range(len(retrieved_results)):
            histories = []

            query = user_queries[i]
            retrieved_result = retrieved_results[i]
            
            prompt = f"Question: {query}\n\nRelated documents: "
            
            for j in range(top_k):
                prompt += ("\n" + f"{j}. " + retrieved_result)
            
            prompt += "\n\nGenerate a answer for me."

            # Call generating function to get generated answer.
            generation = generate_with_loop(prompt, histories)

            answer = ""
            
            # Keep update answer until the whole answer has been generated.
            for ans in generation:
                answer = ans
            
            json_results[i]["answer"] = answer
        
        json.dump(json_results, output_file, indent=4)
        output_file.close()
