import gradio as gr
from llama_cpp import Llama
from config import Config
import pickle
import numpy as np
from config import Config
import json
import time
import faiss
import requests

from threading import Thread

import getopt
from prompt import build_prompt
import sys
from vector_db_manager import Vector_DB, Vector_DB_Faiss, Vector_DB_Qdrant

def query(Inputs, k_vector, max_tokens, temperature, seed):
    if conf.use_rag :
        if conf.external_llama_cpp_url is None:
        # Use internal Llama CPP
            question_embeddings = np.array([llm.embed(Inputs)])
        else :
            api_url = f"{conf.external_llama_cpp_url}/embedding"
            in_data = {"content": Inputs}
            headers = {"Content-Type": "application/json"}
            # print(f"sending : {in_data}")
            response = requests.post(api_url, data=json.dumps(in_data), headers=headers)
            question_embeddings = np.array([json.loads(response.text)['embedding']])
        # toto = self.index.search(question_embeddings,  k=3, distances=0.3)
        retrieved_chunk=db.search(question_embeddings, k_vector)

        str_chunks = ""
        for chunk in retrieved_chunk:
            str_chunks = f"{str_chunks}{chunk}\n--\n"

        prompt = build_prompt(conf.prompt_template, Inputs, str_chunks)
    else :
        prompt = Inputs

    print(prompt)

    start_t = time.time()
    if conf.external_llama_cpp_url is None:
        #Use internal Llama CPP
        output = llm(
            prompt,  # Prompt
            max_tokens=max_tokens,
            # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["Query:", "Question:"],  # Stop generating just before the model would generate a new question
            echo=False,  # Echo the prompt back in the output
            temperature=temperature,
            seed=seed,
        )  # Generate a completion, can also call create_completion
        # print(output)

        prompt_tokens_per_sec = int(output['usage']['prompt_tokens']) / (end_t - start_t)
        completion_tokens_per_sec = int(output['usage']['completion_tokens']) / (end_t - start_t)
        response_text = output['choices'][0]['text']
    else :
        # User external llama cpp server
        api_url = f"{conf.external_llama_cpp_url}/completion"
        in_data = {"prompt": prompt, "n_predict": max_tokens, "seed": seed, "temperature" : temperature}

        # api_url = f"{url}/embedding"
        # in_data = {"content": prompt}

        headers = {"Content-Type": "application/json"}
        # print(f"sending : {in_data}")
        response = requests.post(api_url, data=json.dumps(in_data), headers=headers)
        response_text = json.loads(response.text)['content']
        print(json.loads(response.text))

    end_t = time.time()
    # self.text_ctrl.AppendText("\n\n" + output['choices'][0]['text'])



    return response_text


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

    if conf.external_llama_cpp_url is None :
        llm = Llama(
            model_path=conf.model_path,
            embedding=True,
            n_gpu_layers=conf.n_gpu_layers, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            n_ctx=conf.n_ctx,  # Uncomment to increase the context window
            n_threads=12,
        )

    if conf.use_rag :
        if conf.use_qdrant :
            db = Vector_DB_Qdrant(conf, None)
        else :
            db = Vector_DB_Faiss(conf, None)
        db.load()


    demo = gr.Interface(
        allow_flagging='never',
        fn=query,
        inputs=[gr.Textbox(label="Inputs", lines=10),
                gr.Number(2, label="k"),
                gr.Number(512, label="Max tokens"),
                gr.Number(0.8, label="temperature", step=0.2),
                gr.Number(-1, label="seed", step=1)
                ],
        outputs=[gr.Textbox(label="Outputs", lines=30)],
    )

    demo.launch(server_name=conf.listen_bind, server_port=conf.listen_port)
    # auth=("admin", "pass1234")
