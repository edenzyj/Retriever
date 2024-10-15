from tika import parser
import glob
from langchain.text_splitter import CharacterTextSplitter

pdf_dir = 'pdf/strawberry_file/EN'

file_names = glob.glob(pdf_dir + "/*.pdf")

text_dir = 'embedding_finetune/'

text_file = text_dir + 'contents_EN.txt'

lengths = []

for file_name in file_names:
    # print(file_name)
    text = parser.from_file(file_name)
    pdf_str = text["content"]
    new_str = ""
    
    for i in range(len(pdf_str)):
        if pdf_str[i] == '-' and pdf_str[i+1] == '\n':
            i = i + 1
            continue
        if pdf_str[i] != '\n':
            new_str = new_str + pdf_str[i]
            continue
        if len(new_str) == 0:
            continue
        if pdf_str[i-1] == '.':
            first_letter = str(new_str.split(' ')[-1][0])
            if not first_letter.isupper():
                new_str = new_str + pdf_str[i]
            else:
                new_str = new_str + ' '
        elif i < len(pdf_str) - 1 and pdf_str[i+1].isupper():
            new_str = new_str + pdf_str[i]
        elif new_str[-1] != ' ':
            new_str = new_str + ' '
    
    new_list = new_str.split('\n')
    
    with open(text_dir + '{}.txt'.format(file_name.split('\\')[-1].split('.pdf')[0]), 'w') as fw:
        for split_str in new_list:
            if len(split_str) > 30:
                fw.write('\n')
                lengths.append(len(split_str))
            elif len(lengths) > 0:
                lengths[-1] = lengths[-1] + len(split_str)
            fw.write(split_str)

print(max(lengths))

num = 0

for length in lengths:
    if length > 2000: num = num + 1

print(num)
print(len(lengths))
