from embedding_model import My_embedding

from config import Config
import os
# from llama_cpp import Llama
import numpy as np
import pickle
import getopt
import sys
import requests
import json
from tqdm import tqdm
from vector_db_manager import Vector_DB, Vector_DB_Qdrant
from llm_wrapper import Llm_wrapper

def process_file_md_alt(conf : Config, filename : str) -> [str] :
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

def process_file_md(conf : Config, model : Llm_wrapper, filename : str, max_tokens=256) -> [str] :

    chunks = []
    pre_text = os.path.basename(filename).split('.')[0]
    with open(filename, "r", encoding='utf-8') as f:
        section_pre_text=""
        line = f.readline()
        while not line.startswith('# ') :
            line = f.readline()
        pre_text = line[:-1].lower()
        current_chunk = f"{pre_text} / "
        line = f.readline()
        while line :

            if line.startswith("#") :
                if len(current_chunk) > 100:
                    chunks.append(current_chunk)
                section_pre_text = line[:-1].lower() + " / "
                current_chunk = f"{pre_text} / {section_pre_text} "
            else :
                if line.replace(' ', '') == '\n' :
                    # tokens = model.tokenize(current_chunk)
                    word_count = len(current_chunk.split(' '))
                    if word_count>max_tokens:
                        if len(current_chunk) > 100:
                            chunks.append(current_chunk)
                            current_chunk = f"{pre_text} / {section_pre_text} "
                else :
                    current_chunk += line
            line = f.readline()
        if len(current_chunk) >  len(pre_text) +len (section_pre_text) +20 :
            chunks.append(current_chunk)
    return chunks

def process_file_md_whole(conf : Config, model : Llm_wrapper, filename : str, max_tokens=256) -> [str] :
    chunks = []
    pre_text = os.path.basename(filename).split('.')[0]
    with open(filename, "r", encoding='utf-8') as f:
        section_pre_text = ""
        line = f.readline()
        while not line.startswith('# ') :
            line = f.readline()
        pre_text = line[:-1].lower()
        current_chunk = f"{pre_text} / "

        line = f.readline()
        while line :
            if line.startswith("#") :
                section_pre_text = line[:-1].lower() + " / "
                if len(current_chunk) > len(pre_text) +5 :
                    chunks.append(current_chunk)
                current_chunk = f"{pre_text} / {section_pre_text}"
            elif not line.startswith("#") :
                current_chunk += line.replace('\n', ' ').lower()
            line = f.readline()


        if len(current_chunk) >  len(pre_text)  +len (section_pre_text) +20 :
            chunks.append(current_chunk)
        # if len(chunks) > 2 :
        #     print(chunks)
    return chunks


def process_file_text(conf : Config, model : Llm_wrapper, filename : str, max_tokens=256) -> [str] :
    chunks = []

    with open(filename, "r", encoding='utf-8') as f:

        line = f.readline()
        current_chunk = f""
        while line :
            if len(current_chunk) > 1 :
                tokens = model.tokenize(current_chunk)
                if len(tokens)>max_tokens:
                    if len(current_chunk) > 100:
                        chunks.append(current_chunk)
                        current_chunk = f""
            current_chunk += line
            line = f.readline()
        if len(current_chunk) > 10 :
            chunks.append(current_chunk)
    return chunks
def process_file_basic(conf : Config, filename : str) -> [str] :
    chunks = []
    with open(filename, "r", encoding='utf-8') as f:
        lines = f.read()
    chunk_size = 1000
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    return chunks

def process_file(conf : Config, model : Llm_wrapper, filename : str)  :

    if conf.chunks_mode == 'md' :
        chunks = process_file_md(conf, model, filename, max_tokens=conf.ingest_max_tokens)
    elif conf.chunks_mode == 'md_whole':
        chunks = process_file_md_whole(conf, model, filename, max_tokens=conf.ingest_max_tokens)
    elif conf.chunks_mode == 'text' :
        chunks = process_file_text(conf, model, filename, max_tokens=conf.ingest_max_tokens)

    else :
        chunks = process_file_basic(conf, filename)
    emb_chunks = []

    return  chunks

if __name__ == '__main__':
    conf_file_name = "config.json"

    opts, args = getopt.getopt(sys.argv[1:],"hc:")
    for opt, arg in opts:
        if opt == '-h':
            print(sys.argv[0] + ' -c <conf_file>')
            sys.exit()
        elif opt in ("-c"):
            conf_file_name = arg

    conf = Config(conf_file=conf_file_name)

    llm = Llm_wrapper(conf, only_embd=True)


    input_files = [os.path.join(conf.ingest_files_dir, f) for f in os.listdir(conf.ingest_files_dir)]
    input_files = sorted(input_files)
    print(f"Total number of files {len(input_files)}")

    if conf.ingest_limit_files is not None and conf.ingest_limit_files > 0 :
        input_files = input_files[conf.ingest_start_file_index:conf.ingest_start_file_index+conf.ingest_limit_files]

    emb = My_embedding("path_finder_emb.pk")

    stack = []
    stack_chunks = []
    print("Read file")
    for i in tqdm(range(len(input_files))) :
    # for filename in input_files :
        filename = input_files[i]
        chunks = process_file(conf, llm, filename)
        # stack.append(text_embeddings)
        stack_chunks.extend(chunks)
        emb.add_train_example(chunks, label=float(i))
        # print(text_embeddings)


    # print(stack_chunks)
    print(len(stack_chunks))
    # for s in stack_chunks :
    #     print(s)

    # emb.fit(epochs=200, batch_size=10, warmup_steps=100)
    # emb.save("path_finder_emb.pk")

    # stack_chunks= stack_chunks[:5]
    
    idx = 0
    array_emb = []
    # for chunk in stack_chunks :
    print("Compute Embeddings")
    db = None
    for i in tqdm(range(len(stack_chunks))) :
        chunk = stack_chunks[i]
        array_emb.append(llm.embed(chunk))
        # array_emb.append(emb.encode([chunk])[0])

        # if len(array_emb) > 300 :
        #     text_embeddings = np.array(array_emb)
        #     if db == None :
        #         d = text_embeddings.shape[1]
        #         print(f"dim : {d}")
        #         db = Vector_DB_Qdrant(conf, d)
        #         db.reset()
        #     db.add(text_embeddings, stack_chunks)
        #     array_emb = []

        idx +=1
    text_embeddings = np.array(array_emb)

    if db == None:
        d = text_embeddings.shape[1]
        print(f"dim : {d}")
        db = Vector_DB_Qdrant(conf, d)
        db.reset()

    with open('embeded.pkl', 'wb') as file:
        pickle.dump((text_embeddings, stack_chunks), file)

    db.add(text_embeddings, stack_chunks)
    db.save()

