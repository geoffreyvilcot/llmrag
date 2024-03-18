# llmrag

## Presentation

This repository offers a light solution to exepriment a LLM RAG on your computer. 

## Installation

Create a python virtual environment.

```
python -m venv venv
```

Install llama-cpp-python with your accceleration option, see : https://github.com/abetlen/llama-cpp-python

For Metal:
```
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

For Cuda:
```
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

For reinstal and rebuild
```
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

Cuda for windows (cmd.com)
```
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

Install other requirements:
```
pip install -r requirements.txt
```

## Models

Since the solution is based on llama.cpp (https://github.com/ggerganov/llama.cpp). Every compatible model can be used. For eample : https://huggingface.co/ikawrakow

## Configuration file

*llmrag* uses a json config file. See sample _config.json_

| Parameter    | Description                                                                                                                              | Example                 |
|--------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| model_path   | File path for the model                                                                                                                  | "models/mistral.gguf"   |
| ingest_files_dir  | Directory path where the files to ingest are present                                                                                     | "docs"                  |
| ingest_start_file_index | For partial ingestion, index of the first file to begging with.                                                                          | 20                      |
| ingest_limit_files| For partial ingestion, maximum number of files to process.                                                                               | 10                      |
| use_rag  | Boolean if the RAG functionnality must be used.                                                                                          | true                    |
| vector_db_file | Path for the local db Path (faiss or Qdrant)                                                                                             | "vector_db/demo.db"     |
| use_qdrant | Boolean if Qdrant must be used as the Vector DB, Faiss otherwise                                                                         | true                    |
| qdrant_local | Boolean if Qdrant must be embedded in local                                                                                              | false                   |
| qdrant_host | Hostname for Qdrant server. Only if _qdrant_local_ is true                                                                               | "localhost"             |
| qdrant_port | TCP port for Qdrant server. Only if _qdrant_local_ is true                                                                               | 6333                    |
| qdrant_collection | Collection name port for Qdrant server. Only if _qdrant_local_ is true                                                                   | "demo"                  |
| n_ctx | Context size for _llama.cpp_                                                                                                             | 2048                    |
| n_gpu_layers | Number of layers to be offloaded on the GPU, -1 for all                                                                                  | -1                      |
| external_llama_cpp_url | (Optional) Url for _llama.cpp_. If _None_ or empty. Use local _llama.cpp_                                                                | "http://127.0.0.1:8080" |
| external_llama_cpp_api_key | (Optional) Api Key for _llama.cpp_                                                                                                       | "secretkey"          |   |
| chunks_mode | Mode to use for document splitting.                                                                                                      | "text", "md" or "basic" |
|ingest_max_tokens | Max tokens count for document splitting, Only for "md" of "text". NB: the real count can be slightly higher to avoid cut line in middle. | 512                     |
 | prompt_template | Path to the file with the tempalte to use in Rag mode inference                                                                          | "prompt_default.txt"    |
|listen_bind | Binding host for _webgui_. Set "0.0.0.0" to open Gradio Gui on the network                                                               | "127.0.0.1"            |
| listen_port | Tcp port to used for Gradio.                                                                                                             | 49283                   |



## Template

To generate RAG promte, a template is used.

A template must have:
* *{retrieved_chunk}* string for the chunk text to insert
*  *{input_text}* string for the query wrote by the user.

For example:
```
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {input_text}
Answer:
```

NB: The template file is read for each query, the server does not need to be restarted to take in account modifications.

## Usage

### Ingestion

Create a folder with your data to ingest. Currently only text files are supported.

Indicates the folder under the parameter _ingest_files_dir_ in your configuration file.

Run `python ingest.py [-c config_file_path]`

### Graphical User Interface

A Web interface in _Gradio_ is available : `python webgui.py [-c config_file_path]`

### Webservice
 [Currently not working]
A rest api can be used.

First launch `python webservice.py [-c config_file_path]`.

Usage example:
```
curl -x POST -H 'Content-Type: application/json' -d '{"msg": "Your Query", "max_tokens" : 256}' http://127.0.0.1:16080
```

## Qdrant tips

List all collections
```
curl http://<host>:6333/collections
```

Delete a collections
```
curl -X "DELETE" http://<host>:6333/collections/<collection_name>
```