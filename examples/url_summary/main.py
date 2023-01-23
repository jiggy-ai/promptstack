# summarize a given URL


from promptstack import CompletionTask, ModelConfig, CompletionLimits, OPENAI_BACKEND
from extract import url_to_text
import urllib.parse


class UrlSummaryTask(CompletionTask):

    """
    A Factory class to dynamically compose a prompt context for URL Summary task
    """
    
    def __init__(self):
        model_config = ModelConfig(model       = 'text-davinci-003', # model name
                                   temperature = 0.1, # model sampling temperature in [0,1]
                                   top_p       = 1) # sample from top_p cummulative token probability only

        limits = CompletionLimits(min_prompt     = 0,
                                  min_completion = 300,
                                  max_completion = 600,
                                  max_context    = 4097)

        super().__init__(backend = OPENAI_BACKEND,
                         limits  = limits,
                         config  = model_config)


    def summarize_url(self, url: str):
        """
        return a summary for the given url. 
        """

        # The url is required in able to enable host-specific prompt strategy.
        # For example a different prompt is used to summarize github repo's versus other web sites.
        # the PROMPT_PREFIX is prepended to the url content before sending to the language model
        
        if urllib.parse.urlparse(url).netloc == 'github.com':
            prefix = self.GITHUB_PROMPT_PREFIX
        else:
            prefix = self.SUMMARIZE_PROMPT_PREFIX
        
        url_text = url_to_text(url)
        prompt = prefix + url_text
        prompt.truncate(self.max_prompt_tokens())

        print(f"prompt tokens: {len(prompt)}")

        completion = self.completion(prompt)
        
        print("\n\n******Completion******")
        print(str(completion))
    
task = UrlSummaryTask()

url = 'https://gist.github.com/yoavg/59d174608e92e845c8994ac2e234c8a9'
task.summarize_url(url)

