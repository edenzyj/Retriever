import sys

sys.path.append("..")

from pcs_llama3.llama3_local import get_llm, get_tokenizer, get_sampling_params

def generate_with_loop(message, histories):
    history = []
    
    for human, assistant in histories:
        history.append({"role": "user", "content": human })
        history.append({"role": "assistant", "content": assistant})
    history.append({"role": "user", "content": message})
    
    llm = get_llm()
    tokenizer = get_tokenizer()
    sampling_params = get_sampling_params()
    
    prompt = tokenizer.apply_chat_template(history, tokenize=False)
    
    for chunk in llm.generate(prompt, sampling_params):
        yield chunk.outputs[0].text

if __name__ == "__main__":
    user_query = "What is Anthracnose caused by?"
    
    result = user_query + " " + "All three species also cause leaf spots (Howard et al., 1992; Maas and Palm, 1997; Smith, 1998c)."
    
    histories = ""
    
    generated_answer = generate_with_loop(result, histories)
    
    answer = ""
    
    for ans in generated_answer:
        answer = ans
    print(answer)
