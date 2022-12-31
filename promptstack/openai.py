# GPT3 specific Completions
# Copyright (C) 2022 William S. Kish

import os
from loguru import logger
from time import sleep

from promptstack import *
import openai
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




class OpenAIModelTask(ModelTask):
    """
    Base class for configured model backend implementation of a tokenizer and completion
    This is implemented by a model-backend specific class and returned by ModelBackend.model_task()

    User doesn't usually need to worry about this as this is used internally to base CompletionTask.
    """
    RETRY_COUNT : int  = 5
    config      : ModelConfig

    def token_len(self, text : str) -> int:
        """
        model specific tokenizer
        return the number of model-specific tokens in the text string
        """
        return len(tokenizer(text)['input_ids'])

    
    def completion(self, prompt : str, max_completion_tokens : int) -> str:
        """
        return the completion string for the specified prompt, subject to max_tokens limit
        """
        def completion():
            resp = openai.Completion.create(prompt      = prompt,
                                            max_tokens  = max_completion_tokens,                                            
                                            engine      = self.config.model,
                                            temperature = self.config.temperature,
                                            top_p       = self.config.top_p,
                                            stop        = self.config.stop)

            return resp.choices[0].text
            
        for i in range(OpenAIModelTask.RETRY_COUNT):
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

    
    def subprompt(self,
                  text: str,
                  max_tokens=None,
                  truncate=False,
                  precise=False) -> SubPrompt:
        """
        return a SubPrompt configured with model-specific tokenizer
        """
        return SubPrompt(text           = text,
                         max_tokens     = max_tokens,
                         truncate       = truncate,
                         precise        = precise,
                         token_len_func = self.token_len)


        
class OpenAIBackend(ModelBackend):
    name = "OpenAI-API"

    def model_task(self, config : ModelConfig) -> ModelTask:
        return OpenAIModelTask(config)
    
    def __init__(self):
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        except:
            raise Exception("Set OPENAI_API_KEY environment variable to your OPENAI API Key")

        assert(config.model in OPENAI_COMPLETION_MODELS)        
        super().__init__()

    

