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

# Directory/File name of non-formal content.
reference_directory = "purify/reference_paper/"
keywords_file = "purify/keywords.txt"

# Directory name of both retrieved results file and generated answers file.
output_directory = "results/97/"

# File name of retrieved results file.
result_file = "9907_tart_nFT_Llama3.2_1000Q_k10_Rtv_woKeyword.json"

# The number of retrieved results merged.
top_k = 10

# Use reranker or not.
# If use reranker, fill in the model name in string type.
# If not use reranker, fill in None
reranker = "facebook/tart-full-flan-t5-xl"

# File name of generated answers file.
answer_file = "9907_tart_nFT_Llama3.2_1000Q_k10_Ans_woKeyword.json"
