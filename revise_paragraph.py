from selenium.webdriver.remote.webdriver import By
import undetected_chromedriver as uc
from selenium.webdriver.common.keys import Keys

import time
import glob

class gptParser:
    def __init__(self,
                 driver,
                 gpt_url: str = 'https://chatgpt.com/'):
       
        # Start a webdriver instance and open ChatGPT
        self.driver = driver
        self.driver.get(gpt_url)

    @staticmethod
    def get_driver():
        uc.TARGET_VERSION = 124
        
        options = uc.ChromeOptions()
        options.add_argument("--incognito")

        driver = uc.Chrome(options=options)
        
        return driver

    def __call__(self, msg: str):
        # Find the input field and send a question
        input_field = self.driver.find_elements(By.ID, 'prompt-textarea')[0]
        input_field.send_keys(msg)
        time.sleep(5)
        send_button = self.driver.find_element(By.CSS_SELECTOR, '[data-testid="send-button"]')

        # Press (click) the button
        send_button.click()
        
        # previous information
        # 'p'：text, 'code':code
        try:
            all_elements = self.driver.find_elements(By.CSS_SELECTOR, "code, p")
            # Arrange the text and code in order
            indexed_elements = list(enumerate(all_elements))
            sorted_elements = sorted(indexed_elements, key=lambda x: x[0])
            self.history = [ele.text for idx, ele in sorted_elements]
            self.history.remove('ChatGPT')
        except:
            self.history = []

    def read_respond(self):
        try:
            l = []
            all_elements = self.driver.find_elements(By.CSS_SELECTOR, "code, p")
            indexed_elements = list(enumerate(all_elements))
            sorted_elements = sorted(indexed_elements, key=lambda x: x[0])
            # only return the newest information
            for i in range(len(self.history), len(sorted_elements)-1):
                response = sorted_elements[i][1].text
                l.append(response)
            return l
        except:
            return None

    def new_chat(self):
        self.driver.find_elements("class name", 'text-token-text-primary')[3].click()

    def close(self):
        self.driver.quit()
        
def generate_revise_paragraph(paragraph, fw):
    driver = gptParser.get_driver()
    gpt_parser = gptParser(driver)
    
    query = "I'll give you some contents from an article later. There might be some website links in the contents, and please delete them.  Also, There might be some typos, wrong sentences order, and lost words/characters. Please reply me the revised texts including one paragraph or several paragraphs in utf-8. If the context includes a table, please detailly describe the table in a paragraph instead of pasting it. The paragraph is shown below: {} Just give me the revised context without any other words.".format(paragraph)
    
    time.sleep(3)
    gpt_parser(query)
    
    time.sleep(10)
    response = gpt_parser.read_respond()
    
    for i in range(len(response)):
        fw.write(response[i])
        fw.write("\n")
    
    time.sleep(10)
    
    driver.close()

    return

paragraph_dir = "embedding_finetune/paragraphs/"
file_names = glob.glob(paragraph_dir + '*.txt')

revised_dir = "embedding_finetune/revised_paragraphs/"

if __name__ == "__main__":
    for file_name in file_names:
        with open(file_name, 'r') as fr:
            with open(revised_dir + '{}_revised.txt'.format(file_name.split('\\')[1].split('.txt')[0]), 'w') as fw:
                paragraphs = fr.readlines()
                input_text = ""
                for i in range(len(paragraphs)):
                    paragraph = paragraphs[i].split("\n")[0]
                    if len(input_text) + len(paragraph) < 4000:
                        input_text = input_text + " " + paragraph
                    else:
                        generate_revise_paragraph(input_text, fw)
                        input_text = paragraph
                generate_revise_paragraph(input_text, fw)
