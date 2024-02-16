import random

import copy
import time
import sys
import pickle
from config import Config
import json
from threading import Thread
from flask import Flask, request, send_file
import logging
from flask.logging import default_handler
import urllib
import numpy as np
import getopt
from llama_cpp import Llama
from prompt import build_prompt

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.logger.removeHandler(default_handler)

force_stop = False

@app.route('/llmrag/query', methods=['POST'])
def llmrag_query():
    msg = str(json.loads(request.data.decode())['msg'])

    max_tokens = 10
    if "max_tokens" in json.loads(request.data.decode()):
        max_tokens = int(json.loads(request.data.decode())['max_tokens'])


    question_embeddings = np.array([llm.embed(msg)])
    D, I = index.search(question_embeddings, k=2)  # distance, index
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

    prompt = build_prompt(conf.prompt_template, msg, str(retrieved_chunk))

    print(prompt)

    start_t = time.time()
    output = llm(
        prompt,
        max_tokens=max_tokens,
        stop=["Query:", "Question:"],
        echo=False
    )
    end_t = time.time()
    print(output)
    response = output['choices'][0]['text']

    prompt_tokens_per_sec = int(output['usage']['prompt_tokens']) / (end_t - start_t)
    completion_tokens_per_sec = int(output['usage']['completion_tokens']) / (end_t - start_t)
    return response

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
    with open(conf.vector_db_file, 'rb') as file:
        index, chunks = pickle.load(file)

    app.run(host="0.0.0.0", port=16080)