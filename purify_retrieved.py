import os
import json

from generation import generate_with_loop

if __name__ == "__main__":
    # =====Setting Here=====
    # Directory name and file name of purified file.
    retrieved_dir = "results/97/"
    purified_file = "9907_RR100_nFT_Llama32_1000Q_k10_noReference.json"
    top_k = 10

    with open(retrieved_dir + purified_file, 'r') as fr:
        data = json.load(fr)
        fr.close()

    num = 0
    
    # Generate one answer for modified prompt (question + retrieved context).
    for id, item in enumerate(data):
        if len(item['retrieved_context']) is not top_k:
            query = item['query']
            prompt = f"Question: {query} \n\nRelevant documents: "
            for context in item['retrieved_context']:
                prompt += (context + "\n")
            prompt += "\nAnswer: "

            histories = []
            generation = generate_with_loop(prompt, histories)

            answer = ""
            for ans in generation:
                answer = ans

            item['Answer'] = answer

            num += 1

    print(num)
    
    # Write answers to purified file.
    with open(retrieved_dir + purified_file, "w") as fw:
        json.dump(data, fw, indent=4)
        fw.close()
