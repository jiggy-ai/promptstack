# Promptstack quickstart


from promptstack import *
from random import sample



class StoryTask(CompletionTask):

    PROMPT_PREFIX = "Tell me a story about the following animals:"

    animals = ['cow', 'horse', 'pig', 'dog', 'cat']

    def make_story(self):
        prompt = self.subprompt(self.PROMPT_PREFIX)

        for animal in sample(self.animals, 2):
            prompt += animal

        print(f"prompt tokens: {len(prompt)}")

        completion = self.completion(prompt)
        print(str(completion))
            


    def __init__(self):
        limits = CompletionLimits(min_prompt=10,
                                  min_completion=400,
                                  max_completion=800,
                                  max_context=4097)
        
        super().__init__(backend = OPENAI_BACKEND,
                         limits = limits,
                         config = ModelConfig(model='text-davinci-003'))
        
            
task = StoryTask()


task.make_story()
print()
task.make_story()
                 
