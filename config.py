# The path of vector DB.
database_path = "vectorDB_9907_nFT"

# The path of enterprise data.
paragraph_directory = 'enterprises/revised_97'

# The chunk size used by text splitter.
chunk_size = 200

# Embedding model
# The embedding model repo from HuggingFace or model path from local.
embedding_model_path = 'dunzhang/stella_en_1.5B_v5'
# Use finetuned embedding model or not.
use_finetuned_model = False

# Directory name and file name of query file.
query_directory = "questions/"
query_file = "queries_1000.txt"

# Directory name of non-formal content.
reference_directory = "purify/reference_paper/"

# Directory name of both retrieved results file and generated answers file.
output_directory = "results/97/"

# File name of retrieved results file.
<<<<<<< HEAD
result_file = "9907_tart_nFT_Llama3.2_1000Q_10th_Rtv_test.json"
=======
result_file = "9907_tart_nFT_Llama3.2_1000Q_10th_pure_Rtv.json"
>>>>>>> bd27009f3c29175c0d59d998b24ea7946ad35510

# The number of retrieved results merged.
top_k = 10

# Use reranker or not.
# If use reranker, fill in the model name in string type.
# If not use reranker, fill in None
reranker = "facebook/tart-full-flan-t5-xl"

# File name of generated answers file.
<<<<<<< HEAD
answer_file = "9907_tart_nFT_Llama3.2_1000Q_10th_Ans_test.json"
=======
answer_file = "9907_tart_nFT_Llama3.2_1000Q_10th_pure_Ans.json"
>>>>>>> bd27009f3c29175c0d59d998b24ea7946ad35510
