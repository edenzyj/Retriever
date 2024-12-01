import os

from generation import generate_with_loop

if __name__ == "__main__":
    # =====Setting Here=====
    # Directory name and file name of query file (input file).
    query_dir = "questions/"
    query_file = "questions_100.txt"

    with open(query_dir + query_file, 'r') as fr:
        user_queries = fr.read().split("\n")

    # =====Setting Here=====
    # Directory name of results and file name of answer file (output file).
    result_dir = "results/naive/"
    answer_file = "Llama3-8b_100Q_1st_Ans.txt"

    # Check whether the directory of results exists.
    # If the directory of results does not exist, create it.
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # Generate one answer for each query(question) and write them into answer file.
    with open(result_dir+answer_file, "w") as output_file:
        for i in range(len(user_queries)):
            histories = []
            
            # Generate an answer through llm.
            generation = generate_with_loop("Here is a question : " + user_queries[i] + " Generate a answer for me.", histories)

            answer = ""

            # Keep update answer until the whole answer has been generated.
            for ans in generation:
                answer = ans
            
            output_file.write("Answer {} :".format(i))  # Give a prefix to let all answers easier to be seperated when we do comparison later.
            output_file.write("\n")
            output_file.write(answer)
            output_file.write("\n")
            output_file.write("\n")
