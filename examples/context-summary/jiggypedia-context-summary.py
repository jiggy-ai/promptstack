from contextsummarytask import ContextSummaryTask

from jiggypedia import search

task = ContextSummaryTask()

#query = 'When was the naval ship Moskva sunk and who sunk it?'

query = 'List the different Russian naval ships named Moskva including the years in which they were commissioned and their final disposition.'

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


for r in search(query, k=50, max_item_tokens=500):

    if r.token_count > 500:
        print("filter", r.name, r.token_count)
        continue
    for text in token_chunks(r.text, 3000):
        summary = task.summary(query, text)

        if "N/A" in summary.text:
            print (f"{r.name} N/A")
            continue
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

