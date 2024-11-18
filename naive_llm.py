import os

from generation import generate_with_loop

if __name__ == "__main__":
    query_dir = "questions/"
    query_file = "questions_100.txt"

    with open(query_dir + query_file, 'r') as fr:
        user_queries = fr.read().split("\n")

    result_dir = "results/naive/"

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    result_file = "Llama3-8b_100Q_1st_Ans.txt"

    with open(result_dir+result_file, "w") as output_file:
        for i in range(len(user_queries)):
            histories = ""
            
            generation_reranker = generate_with_loop("Here is a question : " + user_queries[i] + " Generate a answer for me.", histories)

            answer_reranker = ""

            for ans in generation_reranker:
                answer_reranker = ans
            
            output_file.write("Answer {} :".format(i))
            output_file.write("\n")
            output_file.write(answer_reranker)
            output_file.write("\n")
            output_file.write("\n")
