# The path of vector DB.
database_path = "vectorDB"

# The path of enterprise data.
paragraph_directory = 'enterprises/revised_97'

# The chunk size used by text splitter.
chunk_size = 200

# Embedding model
# The embedding model repo from HuggingFace or model path from local.
embedding_model_path = 'dunzhang/stella_en_1.5B_v5'
# Use finetuned embedding model or not.
use_finetuned_model = True

# Directory name and file name of query file.
query_directory = "questions/"
query_file = "questions_100.txt"

# Directory name of both retrieved results file and generated answers file.
output_directory = "results/6/"

# File name of retrieved results file.
result_file = "6_tart_stella1.5B_100Q_1st_Rtv.txt"

# The number of retrieved results merged.
top_k = 5

# Use reranker or not.
# If use reranker, fill in the model name in string type.
# If not use reranker, fill in None
reranker = "facebook/tart-full-flan-t5-xl"

# File name of generated answers file.
answer_file = "6_tart_stella1.5B_100Q_1st_Ans.txt"