# GPT3 specific Completions
# Copyright (C) 2022 William S. Kish

import os
from loguru import logger
from time import sleep


import openai
from .promptstack import *
from .subprompt import SubPrompt

from transformers import GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")



OPENAI_COMPLETION_MODELS = ["text-davinci-003", "text-davinci-002", "text-davinci-001"]

   
    


def GPT3CompletionLimits(min_prompt:int,
                         min_completion:int,
                         max_completion:int,
                         max_context: int = 4097) -> CompletionLimits:
    """
    CompletionLimits specific to GPT3 models with max_context of 4097 tokens
    """
    assert(max_context <= 4097)
    return completion.CompletionLimits(max_context    = max_context,   
                                       min_prompt     = min_prompt,
                                       min_completion = min_completion,
                                       max_completion = max_completion)




TRUNCATED_TOKEN_LEN = len(tokenizer(SubPrompt.TRUNCATED)['input_ids'])

class GPT2SubPrompt(SubPrompt):
    def token_len(self, text : str) -> int:
        """
        model specific tokenizer
        return the number of model-specific tokens in the text string
        """
        return len(tokenizer(text)['input_ids'])
    
    def truncated_token_len(self):
        return TRUNCATED_TOKEN_LEN



RETRY_COUNT = 5

class OpenAIModelTask(ModelTask):
    """
    Base class for configured model backend implementation of a tokenizer and completion
    This is implemented by a model-backend specific class and returned by ModelBackend.model_task()

    User doesn't usually need to worry about this as this is used internally to base CompletionTask.
    """

    config      : ModelConfig

    @classmethod
    def SubPrompt(cls):
        """
        return a SubPrompt Class with model-specific tokenizer
        """
        return GPT2SubPrompt
    
    def completion(self, prompt : str, max_completion_tokens : int) -> str:
        """
        return the completion string for the specified prompt, subject to max_tokens limit
        """
        print(prompt)
        print(max_completion_tokens)
        def completion():
            print(self.config.model)
            
            resp = openai.Completion.create(prompt      = prompt,
                                            max_tokens  = max_completion_tokens,
                                            engine      = self.config.model,
                                            temperature = self.config.temperature,
                                            top_p       = self.config.top_p,
                                            stop        = self.config.stop)

            return resp.choices[0].text
        return completion()
    
        for i in range(RETRY_COUNT):
            try:
                return completion()
            except openai.error.ServiceUnavailableError:
                logger.warning("openai ServiceUnavailableError")
                if i == RETRY_COUNT-1:
                    raise
                sleep((i+1)*.1)
            except Exception as e:
                logger.exception("_completion")
                raise

    


        
class OpenAIBackend(ModelBackend):
    name = "OpenAI-API"

    def model_task(self, config : ModelConfig) -> ModelTask:
        assert(config.model in OPENAI_COMPLETION_MODELS)                
        return OpenAIModelTask(config=config)
    
    def __init__(self):
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        except:
            raise Exception("Set OPENAI_API_KEY environment variable to your OPENAI API Key")
        super().__init__()

    

