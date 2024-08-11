import sys

sys.path.append("../..")

from LLaMA.llama3_local.llama3_local import get_llm, get_tokenizer, get_sampling_params

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
