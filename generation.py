import sys

sys.path.append("..")  # Llama3 model will be outside this repository/directory, so path append is necessary.

# =====Setting Here=====
# Import llama3 from repository pcs_llama3 cloned from https://github.com/IoTtalk/pcs_llama3.git.
from pcs_llama3.llama3_local import get_llm, get_tokenizer, get_sampling_params


def generate_with_loop(message, histories):
    """
    Generate answer according to chat histories and newly given message.

    Parameters:
        message (str): The new query input to generate answer.
        histories (list): The chat histories including contents from both human(user) and assistant(llm).

    Returns:
        str: A generated answer which is remaining fulfilled.
    """
    
    history = []
    
    # Add all chat content from both human and assistant.
    for human, assistant in histories:
        history.append({"role": "user", "content": human})
        history.append({"role": "assistant", "content": assistant})
    # Add message into the list and mark its role as user.
    history.append({"role": "user", "content": message})
    
    # =====Setting Here=====
    # Choose a version of llama3 from HuggingFace.
    llm = get_llm("casperhansen/llama-3-8b-instruct-awq")
    tokenizer = get_tokenizer()
    sampling_params = get_sampling_params()
    
    prompt = tokenizer.apply_chat_template(history, tokenize=False)
    
    # Keep return the newest result generated from llm.
    for chunk in llm.generate(prompt, sampling_params):
        yield chunk.outputs[0].text


# Run this python file independently to try the function defined before.
if __name__ == "__main__":
    user_query = "What is Anthracnose caused by?"
    
    histories = []
    
    # Call function "generate_with_loop" to generate answer using llama3.
    generated_answer = generate_with_loop(user_query, histories)
    
    answer = ""
    
    # Keep update answer until the whole answer has been generated.
    for ans in generated_answer:
        answer = ans
    
    print(answer)
