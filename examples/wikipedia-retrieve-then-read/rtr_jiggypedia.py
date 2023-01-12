# Promptstack quickstart

import wikipedia
from promptstack import *
from jiggypedia import search
import numpy as  np
from bs4 import BeautifulSoup, NavigableString, Tag


class WikipediaSearchTask(CompletionTask):

    
    def query(self, querytext: str) -> str:
        
        prompt = self.PREFIX
        prompt += querytext
        prompt += "Information:"
        remaining_tokens = self.limits.remaining_prompt_tokens(prompt + "Answer:")
        print(f"remaining_tokens: {remaining_tokens}")
        
        results = [r.text for r in search(querytext, k=5)]   
        for chunk in results:
            print(chunk)
            new_remaining_tokens = self.limits.remaining_prompt_tokens(prompt + chunk)
            print(f"new remaining_tokens: {new_remaining_tokens}")  
            if new_remaining_tokens < 1:
                break
            prompt += chunk
            remaining_tokens = new_remaining_tokens

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
                        
    def __init__(self):
        limits = CompletionLimits(min_prompt=200,
                                  min_completion=200,
                                  max_completion=400,
                                  max_context=4097)

        
        super().__init__(backend = OPENAI_BACKEND,
                         limits = limits,
                         config = ModelConfig(model = 'text-davinci-003',
                                              temperature = 0.1))
        

        
            
task = WikipediaSearchTask()


while True:
    query = input('Query: ').rstrip()
    #query = "How is a rocket engined throttled?"

    task.query(query)
