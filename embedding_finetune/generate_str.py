import sys
from tika import parser
import glob

sys.path.append("..")

pdf_dir = 'pdf/strawberry_file/EN'

file_names = glob.glob(pdf_dir + "/*.pdf")

texts = []

with open('sentences_EN.txt', 'w') as fw:
    for file_name in file_names:
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
            if i == 0:
                continue
            if pdf_str[i-1] == '.':
                first_letter = str(new_str.split(' ')[-1][0])
                if not first_letter.isupper():
                    texts.append(new_str)
                    new_str = ""
                continue
        new_list = new_str.split('\n')
        for split_str in new_list:
            texts.append(split_str)

    for text in texts:
        fw.write(text)
        fw.write('\n\n')
