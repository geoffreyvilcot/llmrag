from llama_cpp import Llama
from config import Config
import pickle
import numpy as np

conf = Config(conf_file="config.json")

llm = Llama(
    model_path=conf.model_path,
    embedding=True,
    # n_gpu_layers=-1, # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    # n_ctx=2048, # Uncomment to increase the context window
)


#
# output = llm(
#       "Q: Name the planets in the solar system? A: ", # Prompt
#       max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#       stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#       echo=True # Echo the prompt back in the output
# ) # Generate a completion, can also call create_completion
# print(output)

# res_emb = llm.embed("L’IA générative et son impact sur le quotidien des entreprises")
# print(res_emb)

# Ouvrez le fichier en mode binaire pour lire (rb)
with open(conf.vector_db_file, 'rb') as file:
    index, chunks = pickle.load(file)

question = "générative est une technologie"
question_embeddings = np.array([llm.embed(question)])
D, I = index.search(question_embeddings, k=2)  # distance, index
retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
# print(retrieved_chunk)

prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""

output = llm(
      question, # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Query:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)