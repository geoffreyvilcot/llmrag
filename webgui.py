import gradio as gr
from llama_cpp import Llama
from config import Config
import pickle
import numpy as np
from config import Config
import json
import time
import faiss

from threading import Thread

import getopt
from prompt import build_prompt
import sys
from vector_db_manager import Vector_DB, Vector_DB_Faiss, Vector_DB_Qdrant

def query(Inputs, k_vector, max_tokens):
    question_embeddings = np.array([llm.embed(Inputs)])
    # toto = self.index.search(question_embeddings,  k=3, distances=0.3)
    retrieved_chunk=db.search(question_embeddings)

    str_chunks = ""
    for chunk in retrieved_chunk:
        str_chunks = f"{str_chunks}{chunk}\n"

    prompt = build_prompt(conf.prompt_template, Inputs, str_chunks)

    print(prompt)

    start_t = time.time()
    output = llm(
        prompt,  # Prompt
        max_tokens=max_tokens,
        # Generate up to 32 tokens, set to None to generate up to the end of the context window
        stop=["Query:", "Question:"],  # Stop generating just before the model would generate a new question
        echo=False  # Echo the prompt back in the output
    )  # Generate a completion, can also call create_completion
    end_t = time.time()
    print(output)
    # self.text_ctrl.AppendText("\n\n" + output['choices'][0]['text'])

    prompt_tokens_per_sec = int(output['usage']['prompt_tokens']) / (end_t - start_t)
    completion_tokens_per_sec = int(output['usage']['completion_tokens']) / (end_t - start_t)

    return output['choices'][0]['text']


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

    llm = Llama(
        model_path=conf.model_path,
        embedding=True,
        n_gpu_layers=conf.n_gpu_layers, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        n_ctx=conf.n_ctx,  # Uncomment to increase the context window
    )

    if conf.use_qdrant :
        db = Vector_DB_Qdrant(conf, None)
    else :
        db = Vector_DB_Faiss(conf, None)
    db.load()


    demo = gr.Interface(
        allow_flagging='never',
        fn=query,
        inputs=[gr.Textbox(label="Inputs", lines=10),
                gr.Number(3, label="k"),
                gr.Number(512, label="Max tokens"),
                ],
        outputs=[gr.Textbox(label="Outputs", lines=30)],
    )

    demo.launch(server_name="0.0.0.0", server_port=49283)
    # auth=("admin", "pass1234")
