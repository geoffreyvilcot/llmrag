from llama_cpp import Llama
from config import Config
import requests
import json
import numpy as np

class Llm_wrapper(object) :
    def __init__(self, conf : Config):
        super().__init__()
        self.conf = conf
        self.llm = None
        if conf.external_llama_cpp_url is None:
            self.llm = Llama(
                model_path=conf.model_path,
                embedding=True,
                n_gpu_layers=conf.n_gpu_layers,  # Uncomment to use GPU acceleration
                # seed=1337, # Uncomment to set a specific seed
                n_ctx=conf.n_ctx,  # Uncomment to increase the context window
                verbose=True,
                n_threads_batch=conf.n_threads,
            )
            self.llm.verbose = False
    def embed(self, inputs):
        if self.conf.external_llama_cpp_url is None:
        # Use internal Llama CPP
            return np.array(self.llm.embed(inputs))
        else :
            api_url = f"{self.conf.external_llama_cpp_url}/embedding"
            in_data = {"content": inputs}
            headers = {"Content-Type": "application/json"}
            if self.conf.external_llama_cpp_api_key is not None :
                headers["Authorization"] = f"Bearer {self.conf.external_llama_cpp_api_key}"
            # print(f"sending : {in_data}")
            response = requests.post(api_url, data=json.dumps(in_data), headers=headers)
            return np.array(json.loads(response.text)['embedding'])

    def query(self, inputs, max_tokens, temperature, seed):
        if self.conf.external_llama_cpp_url is None:
            # Use internal Llama CPP
            self.llm.reset()
            output = self.llm(
                inputs,  # Prompt
                max_tokens=max_tokens,
                # Generate up to 32 tokens, set to None to generate up to the end of the context window
                stop=["Query:", "Question:"],  # Stop generating just before the model would generate a new question
                echo=False,  # Echo the prompt back in the output
                temperature=temperature,
                seed=seed,
                stream=True
            )  # Generate a completion, can also call create_completion
            # print(output)

            for e in output:
                # print(e['choices'][0]['text'])
                yield e['choices'][0]['text']

            # return output
            # response_text = output['choices'][0]['text']
        else:
            # User external llama cpp server
            api_url = f"{self.conf.external_llama_cpp_url}/completion"
            in_data = {"prompt": inputs, "n_predict": max_tokens, "seed": seed, "temperature": temperature}

            headers = {"Content-Type": "application/json"}
            if self.conf.external_llama_cpp_api_key is not None:
                headers["Authorization"] = f"Bearer {self.conf.external_llama_cpp_api_key}"
            response = requests.post(api_url, data=json.dumps(in_data), headers=headers)
            response_text = json.loads(response.text)['content']
            # print(json.loads(response.text))
            yield response_text
        # return response_text

    def tokenize(self, inputs):
        if self.conf.external_llama_cpp_url is None:
            return np.array(self.llm.tokenize(inputs.encode('utf8')))
        else :
            # User external llama cpp server
            api_url = f"{self.conf.external_llama_cpp_url}/tokenize"
            in_data = {"content": inputs}

            headers = {"Content-Type": "application/json"}
            if self.conf.external_llama_cpp_api_key is not None:
                headers["Authorization"] = f"Bearer {self.conf.external_llama_cpp_api_key}"
            response = requests.post(api_url, data=json.dumps(in_data), headers=headers)
            return np.array(json.loads(response.text)['tokens'])


