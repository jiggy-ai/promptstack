
# PromptStack 
# Copyright (C) 2022 Jiggy AI

from time import time
from typing import Optional, List
from subprompt import SubPrompt
from exceptions import *

from pydantic import  BaseModel, ValidationError, validator, Field, condecimal

timestamp = condecimal(max_digits=14, decimal_places=3)  # unix epoch timestamp decimal to millisecond precision



class ModelConfig(BaseModel):
    """
    task-specific model configuration
    """
    model       :  str                          # model name
    temperature :  Optional[float]     = 1      # model sampling temperature in [0,1]
    top_p       :  Optional[float]     = 1      # sample from top_p cummulative token probability only
    stop        :  Optional[List[str]] = []     # additional stop sequence strings

    
    
    
class CompletionLimits(BaseModel):
    """
    specification for limits for a CompletionTask:
    min prompt tokens             -- the minimum acceptable prompt size
    min completion tokens         -- the minimum size completion tokens to reserve in the shared context
    max completion tokens         -- the maximum size completion we will allow
    max_context                   -- the maximum prompt+completion context size. Often set to model maximum, but can be lower.
    """
    min_prompt       : int
    min_completion   : int
    max_completion   : int    
    max_context      : int

    def max_completion_tokens(self, prompt : SubPrompt) -> int:
        """
        returns the maximum completion tokens available given the max_context limit
        and the actual number of tokens in the prompt
        raises MinimumTokenLimit or MaximumTokenLimit exceptions if the prompt
        is too small or too big.
        """
        if prompt.tokens < self.min_prompt:
            raise MinimumTokenLimit
        if prompt.tokens > self.max_prompt_tokens():
            raise MaximumTokenLimit
        max_available_tokens = self.max_context - prompt.tokens
        if max_available_tokens > self.max_completion:
            return self.max_completion
        return max_available_tokens

    def max_prompt_tokens(self) -> int:
        """
        return the maximum prompt size in tokens
        """
        return self.max_context - self.min_completion



class CompletionTask(BaseModel):
    """
    A LLM completion task that shares a particular llm configuration and prompt/completion limit structure. 
    This would typically be a base class for an application-specific completion task which would typically
    include task-specific prompts/prompt stacks.
    """
    backend     : ModelBackend
    config      : ModelConfig
    limits      : CompletionLimits

    def __init__(self,
                 backend     : ModelBackend,
                 config      : ModelConfig,
                 limits      : CompletionLimits) -> "CompletionTask":

        self._model_task = backend.model_task(config)
        
        super().__init__(name=name, backend=backend, config=config, limits=limits)
        

    def subprompt(self,
                  text: str,
                  max_tokens=None,
                  truncate=False,
                  precise=False) -> SubPrompt:
        """
        return model-tokenizer specific subprompt
        """
        return self._model_task.subprompt(text=text, max_tokens=max_tokens, truncate=truncate, precise=precise)


    def max_prompt_tokens(self) -> int:
        return self.limits.max_prompt_tokens()

    
    def completion(self, prompt : SubPrompt) -> Completion:
        """
        Prompt the configued model with the specified prompt and return the resulting Completion,
        subject to task configuration and limits
        """
        # check prompt limits and return max completion size to request
        # given the size of the prompt and the configured limits
        max_completion = self.limits.max_completion_tokens(prompt)
        
        # perform the completion inference
        text = self._model_task.completion(prompt                = str(prompt),
                                           max_completion_tokens = max_completion)

        completion = Completion(task        = self,
                                prompt      = str(prompt),
                                text        = text)

        return completion



class Completion(BaseModel):
    """
    An LLM completion, returned by CompletionTask.completion()
    """
    task       : CompletionTask     # the task that created this completion
    prompt     : str                #  the full prompt input
    text       : str                #  the completion text output from the model
    created_at : timestamp = Field(default_factory=time)  

    def __str__(self):
        """
        str(Completion) returns the completion text for convenience
        """
        return self.text




class ModelTask(BaseModel):
    """
    Base class for configured model backend implementation of a tokenizer and completion
    This is implemented by a model-backend specific class and returned by ModelBackend.model_task()

    User doesn't usually need to worry about this as this is used internally to base CompletionTask.
    """
    config : ModelConfig

    def token_len(self, text : str) -> int:
        """
        model specific tokenizer
        return the number of model-specific tokens in the text string
        """
        pass
    
    def completion(self, prompt : str, max_completion_tokens : int) -> str:
        """
        return the completion string for the specified prompt, subject to max_completion_tokens limit
        """
        pass

    def subprompt(self,
                  text: str,
                  max_tokens=None,
                  truncate=False,
                  precise=False) -> SubPrompt:
        """
        return a SubPrompt configured with model-specific tokenizer
        """
        pass


    
class ModelBackend(BaseModel):
    """
    Base class for a model backend. 
    Examples of backend's implemented elsewhere are OpenAI API and HuggingFace Transformers.

    Backends sole purpose is to return instances of ModelTasks via model_task().
    User doesn't usually need to worry about this as it is handled within the base CompletionTask.
    """
    name : str

    def model_task(self, config : ModelConfig) -> ModelTask:
        pass

    
