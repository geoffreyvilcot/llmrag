from config import Config
import os
from llama_cpp import Llama
import numpy as np
import faiss
import pickle

def process_file_md(conf : Config, filename : str) -> [str] :
    chunks = []
    with open(filename, "r", encoding='utf-8') as f:
        prevline = ""
        line = f.readline()
        current_chunk = ""
        while line or prevline:
            if "----" in line :
                if len(current_chunk) > 10:
                    chunks.append(current_chunk)
                current_chunk = prevline
                prevline = ""
            elif prevline != "" :
                current_chunk += prevline
                prevline = line
            else :
                prevline = line
            line = f.readline()
        if len(current_chunk) > 10 :
            chunks.append(current_chunk)
    return chunks

def process_file_basic(conf : Config, filename : str) -> [str] :
    chunks = []
    with open(filename, "r", encoding='utf-8') as f:
        lines = f.read()
    chunk_size = 1024
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    return chunks

def process_file(conf : Config, model : Llama, filename : str)  :

    chunks = process_file_basic(conf, filename)
    emb_chunks = []
    # for chunk in chunks :
    #     print(chunk)
    #     res_emb = llm.embed(chunk)
    #     print(len(res_emb))
    #     emb_chunks.append(res_emb)
    # text_embeddings = np.array(emb_chunks)
    return  chunks

if __name__ == '__main__':
    conf = Config(conf_file="config.json")

    llm = Llama(
        model_path=conf.model_path,
        embedding=True,
        # n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        # n_ctx=2048, # Uncomment to increase the context window
    )
    input_files = [os.path.join(conf.ingest_files_dir, f) for f in os.listdir(conf.ingest_files_dir)]

    stack = []
    stack_chunks = []
    for filename in input_files :
        chunks = process_file(conf, llm, filename)
        # stack.append(text_embeddings)
        stack_chunks.extend(chunks)
        # print(text_embeddings)


    print(stack_chunks)
    print(len(stack_chunks))
    idx = 0
    array_emb = []
    for chunk in stack_chunks :
        print(f"{idx}/{len(stack_chunks)}")
        array_emb.append(llm.embed(chunk))
        idx +=1
    text_embeddings = np.array(array_emb)
    # text_embeddings = np.array([llm.embed(chunk) for chunk in stack_chunks])
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    with open(conf.vector_db_file, 'wb') as file:
        pickle.dump((index,stack_chunks) , file)

