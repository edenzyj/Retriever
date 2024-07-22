from tart.TART.src.modeling_enc_t5 import EncT5ForSequenceClassification
from tart.TART.src.tokenization_enc_t5 import EncT5Tokenizer
import torch
import torch.nn.functional as F
import numpy as np

# load TART full and tokenizer
model = EncT5ForSequenceClassification.from_pretrained("facebook/tart-full-flan-t5-xl")
tokenizer =  EncT5Tokenizer.from_pretrained("facebook/tart-full-flan-t5-xl")

model.eval()

q = "What is the population of Tokyo?"
in_answer = "retrieve a passage that answers this question from Wikipedia"

p_1 = "The population of Japan's capital, Tokyo, dropped by about 48,600 people to just under 14 million at the start of 2022, the first decline since 1996, the metropolitan government reported Monday."
p_2 = "Tokyo, officially the Tokyo Metropolis (東京都, Tōkyō-to), is the capital and largest city of Japan."

# 1. TART-full can identify more relevant paragraph. 
features = tokenizer(['{0} [SEP] {1}'.format(in_answer, q), '{0} [SEP] {1}'.format(in_answer, q)], [p_1, p_2], padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    scores = model(**features).logits
    normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]

print()
print("==========")
print(np.argmax(normalized_scores))
print("----------")
print([p_1, p_2][np.argmax(normalized_scores)])
