from llama_cpp import Llama
from config import Config
import pickle
import numpy as np

conf = Config(conf_file="config_mixtral.json")

llm = Llama(
    model_path=conf.model_path,
    embedding=True,
    # n_gpu_layers=-1, # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    n_ctx=2048, # Uncomment to increase the context window
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

question = """Question: En exploitant l’IA générative, une entreprise analyse de grands ensembles de données pour extraire des informations et des schémas précieux, permettant ainsi une prise de décision basée sur les données. À quel cas d’utilisation commerciale de l’IA cet exemple correspond-il ?
Instruction : Choisissez l'option qui répond le mieux à la question. 
Choix de réponse :

* Amélioration de la visibilité de la marque.

* Exploration de données pour obtenir des informations.

* Automatisation des processus de recherche et développement.

* Optimisation de la chaîne logistique."""
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

prompt = f"""
Le contexte est décrit ci-dessous.
---------------------
{retrieved_chunk}
---------------------
A partir de ces informations de contexte et avec aucun savoir préalable, Répondre à la question.
Question: {question}
Reponse:
"""

print(prompt)

output = llm(
      prompt, # Prompt
      max_tokens=512, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Query:", "Question:"], # Stop generating just before the model would generate a new question
      echo=False # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)