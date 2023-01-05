# Promptstack quickstart

from sentence_transformers import SentenceTransformer
import wikipedia
from promptstack import *
import numpy as  np
from bs4 import BeautifulSoup, NavigableString, Tag


ST_MODEL_NAME   =  'multi-qa-mpnet-base-dot-v1'
st_model        =  SentenceTransformer(ST_MODEL_NAME)



def extract_text_from_html(content):
    soup = BeautifulSoup(content, 'html.parser')
    text = soup.find_all(text=True)
    output = ''
    blacklist = ['[document]','noscript','header','html','meta','head','input','script', "style"]
    # there may be more elements you don't want
    output = []
    for t in text:
        if t.parent.name not in blacklist:
            output.append(t)
    return output


class WikipediaReadTask(CompletionTask):

    
    def query(self, querytext: str) -> str:
        
        qvector =  st_model.encode(querytext)

        for c in self.chunks:
            c.similarity = np.dot(c.embedding, qvector)
            print("~~~~~~~~~~~~~~~~~~~~~~")
            print(c.similarity)
            print(c.text)
            
        self.chunks.sort(key=lambda x:x.similarity, reverse=True)

        prompt = self.PREFIX
        prompt += querytext
        prompt += "Information:"
        remaining_tokens = self.limits.remaining_prompt_tokens(prompt + "Answer:")
        print(f"remaining_tokens: {remaining_tokens}")
        
        use_chunks = []        
        for chunk in self.chunks:
            if chunk.tokens > remaining_tokens:
                break
            use_chunks.append(chunk)
            remaining_tokens -= chunk.tokens + 1

        # return to original chunk order
        use_chunks.sort(key=lambda x: x.ix)

        for chunk in use_chunks:
            prompt += chunk

        print("remaining", self.limits.remaining_prompt_tokens(prompt))
        prompt += "Answer:"
        print("remaining", self.limits.remaining_prompt_tokens(prompt))            
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("PROMPT:")
        #print(prompt)
        print(f"PROMPT TOKENS: {prompt.tokens}")
        completion = self.completion(prompt)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("COMPLETION:")
        print(str(completion))
        print()
        return completion
                        
    def __init__(self, wikipedia_page_title):
        limits = CompletionLimits(min_prompt=200,
                                  min_completion=200,
                                  max_completion=400,
                                  max_context=4097)

        
        super().__init__(backend = OPENAI_BACKEND,
                         limits = limits,
                         config = ModelConfig(model = 'text-davinci-003',
                                              temperature = 0.1))
        
        #chunks = wikipedia.page(wikipedia_page_title).content.split("\n\n")

        htmlParse = BeautifulSoup(wikipedia.page(wikipedia_page_title).html(), 'html.parser')
        chunks = [p.get_text() for p in htmlParse.find_all("p")]

        SubPrompt = self.SubPrompt()
        self.chunks = [SubPrompt(c) for c in chunks]

        for ix, c in enumerate(self.chunks):
            c.embedding = st_model.encode(c.text)
            c.ix = ix

        
            
task = WikipediaReadTask('rocket engine')


while True:
    query = input('Query: ').rstrip()
    #query = "How is a rocket engined throttled?"

    task.query(query)
