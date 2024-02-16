
def build_prompt(template_path : str, query : str, chunks : str) :
    with open(template_path, "r", encoding='utf-8') as f:
        content = f.read()
    content = content.replace('{retrieved_chunk}', chunks)
    content = content.replace('{input_text}', query)
    return content