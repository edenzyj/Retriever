# Set the path of vector DB.
database_path = "vectorDB"

# These are parameters used to build vector DB.
paragraph_directory = 'enterprises/revised_6'
chunk_size = 200
embedding_model_path = 'dunzhang/stella_en_1.5B_v5'

# Directory name and file name of query file.
query_directory = "questions/"
query_file = "questions_100.txt"

# Directory name of both retrieved results file and generated answers file.
output_directory = "results/6/"

# File name of retrieved results file.
result_file = "6_tart_stella1.5B_100Q_1st_Rtv.txt"

# The number of retrieved results merged.
top_k = 5

# File name of generated answers file.
answer_file = "6_tart_stella1.5B_100Q_1st_Ans.txt"
