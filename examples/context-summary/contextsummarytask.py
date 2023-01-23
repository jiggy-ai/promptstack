# summarize content given context

import wikipedia
from promptstack import *
from jiggypedia import search


class ContextSummaryTask(CompletionTask):

    
    def summary(self, query: str, content: str) -> str:
        
        prompt = self.PREFIX
        prompt += f"Query:\n{query}"
        prompt += f"Content:\n{content}"
        prompt += f"Summary:"
        completion = self.completion(prompt)
        return completion
                        
    def __init__(self):
        limits = CompletionLimits(min_prompt=10,
                                  min_completion=200,
                                  max_completion=400,
                                  max_context=4097)

        
        super().__init__(backend = OPENAI_BACKEND,
                         limits = limits,
                         config = ModelConfig(model = 'text-davinci-003',
                                              temperature = 0.1))
        

        
            


