import gradio as gr
from config import Config
import pickle
import numpy as np
from config import Config
import json
import time
import requests

from threading import Thread

import getopt
from prompt import build_prompt
import sys
from vector_db_manager import Vector_DB, Vector_DB_Qdrant
from llm_wrapper import Llm_wrapper

def query(Inputs, k_vector, max_tokens, temperature, seed):
    if conf.use_rag :
        question_embeddings = np.array([llm.embed(Inputs)])
        # question_embeddings = np.array(emb.encode([Inputs]))

        retrieved_chunk=db.search(question_embeddings, k_vector)

        str_chunks = ""
        for chunk in retrieved_chunk:
            str_chunks = f"{str_chunks}{chunk}\n--\n"

        prompt = build_prompt(conf.prompt_template, Inputs, str_chunks)
    else :
        prompt = Inputs

    print(prompt)

    start_t = time.time()

    iterator_result = llm.query(prompt, max_tokens, temperature, seed)
    response_text = ""
    idx = 0
    start_t = None
    for e in iterator_result :
        if start_t is None :
            start_t = time.time()-1
        response_text +=e
        end_t = time.time()
        idx +=1
        yield response_text, f"Elapsed time {end_t -start_t:0.2f}s / {idx} tokens / {idx/(end_t -start_t):0.2f} tokens / sec", prompt

    end_t = time.time()
    # self.text_ctrl.AppendText("\n\n" + output['choices'][0]['text'])

    print(response_text)
    print("="*120)

    return response_text, f"Elapsed time {end_t -start_t:0.2f}s / {idx} tokens / {idx/(end_t -start_t):0.2f} tokens / sec", prompt


if __name__ == "__main__":
    conf_file_name = "config.json"

    opts, args = getopt.getopt(sys.argv[1:],"hc:")
    for opt, arg in opts:
        if opt == '-h':
            print(sys.argv[0] + ' -c <conf_file>')
            sys.exit()
        elif opt in ("-c"):
            conf_file_name = arg

    conf = Config(conf_file=conf_file_name)

    llm = Llm_wrapper(conf)
    # emb = My_embedding("path_finder_emb.pk")

    if conf.use_rag :
        db = Vector_DB_Qdrant(conf, None)
        db.load()


    demo = gr.Interface(
        allow_flagging='never',
        fn=query,
        analytics_enabled=False,
        inputs=[gr.Textbox(label="Inputs", lines=10),
                gr.Number(6, label="k"),
                gr.Number(512, label="Max tokens"),
                gr.Number(0.8, label="temperature", step=0.2),
                gr.Number(-1, label="seed", step=1)
                ],
        # outputs=[gr.Markdown(label="Outputs"), gr.Label(label="Stats"), gr.Text(label="built prompt")],
        outputs=[gr.Textbox(label="Outputs", lines=30), gr.Label(label="Stats"), gr.Text(label="built prompt")],

    )

    demo.launch(server_name=conf.listen_bind, server_port=conf.listen_port)
    # auth=("admin", "pass1234")






















