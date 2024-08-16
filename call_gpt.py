from selenium.webdriver.remote.webdriver import By
import undetected_chromedriver as uc
import time
from selenium.webdriver.common.keys import Keys

class gptParser:
    def __init__(self,
                 driver,
                 gpt_url: str = 'https://chat.openai.com/'):
       
        # Start a webdriver instance and open ChatGPT
        self.driver = driver
        self.driver.get(gpt_url)

    @staticmethod
    def get_driver():
        uc.TARGET_VERSION = 124
        driver = uc.Chrome()
        # driver = webdriver.Chrome(ChromeDriverManager().install())
        # driver = webdriver.Chrome(executable_path="C:/Users/AnnA/Desktop/chromedriver_win32/chromedriver.exe")
        return driver

    def __call__(self, msg: str):
        # Find the input field and send a question
        input_field = self.driver.find_elements(
            By.TAG_NAME, 'textarea')[0]
        input_field.send_keys(msg)
        input_field.send_keys(Keys.RETURN)
        
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
        
driver = gptParser.get_driver()
gpt_parser = gptParser(driver)

query = "test"
gpt_parser(query) 
time.sleep(5)
response = gpt_parser.read_respond() 
for r in response:
    print(r)
    
# new chat
gpt_parser.new_chat()

# send the query
question = "What is Anthracnose caused by?"

answer_one = "The disease cycles of anthracnose on different hosts have similar \
              components (Peres et al., 2005; De Silva et al., 2017); they are \
              generally polycyclic, with splash-borne asexual spores (conidia)"

answer_two = "Anthracnose is one of the most damaging diseases of strawberries, \
              causing wilting and death of transplants in the nursery field. \
              The pathogen known to be responsible for the disease in Korea is"

query = "There is a farmer asking about a question.  The question is : " + question + "\n" + \
        "This is the first answer : " + answer_one + "\n" + "And this is the second answer : " + answer_two + "\n" + \
        "If you are a botanist, tell me which one is more precise."
print(query)

for i in range (3):
    gpt_parser(query)
    time.sleep(5)
    response = gpt_parser.read_respond()
    for r in response:
        print(r)
