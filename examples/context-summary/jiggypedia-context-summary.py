from contextsummarytask import ContextSummaryTask

from jiggypedia import search

task = ContextSummaryTask()

#query = 'When was the naval ship Moskva sunk and who sunk it?'

query = 'How many different Russian naval ships have been named Moskva?'

SubPrompt=task.SubPrompt()

def token_chunks(text: str, tokens : int):
    """
    yield text (assumed to be newline seperated) in chunks of size tokens or smaller.
    """
    lines = text.split('\n')
    current_tokens = 0
    current_lines = []
    for line in lines:
        if current_tokens + len(SubPrompt(line)) > tokens:
            if not current_lines:
                raise Exception("lines too big")
            yield "\n".join(current_lines)
            current_lines = []
            current_tokens = 0
        current_tokens += len(SubPrompt(line))
        current_lines.append(line)
    if current_lines:
        yield "\n".join(current_lines)


for r in search(query, k=20):

    for text in token_chunks(r.text, 3000):
        summary = task.summary(query, text)

        #print("----------------------------------------------")
        print("----------------------------------------------")        
        print(r.name, r.token_count)
        #print("----------------------------------------------")
        print("----------------------------------------------")
        print("                  SUMMARY")
        print("----------------------------------------------")    
        print(summary)
        print("==============================================")
        print()
        print()

