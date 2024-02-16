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
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

Install other requirements:
```
pip install -r requirements.txt
```

## Models

Since the solution is based on llama.cpp (https://github.com/ggerganov/llama.cpp). Every compatible model can be used. For eample : https://huggingface.co/ikawrakow

## Configuration file

*llmrag* uses a json config file. See sample _config.json_

Todo: Format syntax

## Template

Todo

## Usage

### Ingestion

Create a folder with your data to ingest. Currently only text files are supported.

Indicates the folder under the parameter _ingest_files_dir_ in your configuration file.

Run `python ingest.py [-c config_file_path]`

### Graphical User Interface

A lightweight interface in _wxPython_ is available : `python gui.py [-c config_file_path]`

### Webservice

A rest api can be used.

First launch `python webservice.py [-c config_file_path]`.

Usage example:
```
curl -x POST -H 'Content-Type: application/json' -d '{"msg": "Your Query", "max_tokens" : 256}' http://127.0.0.1:16080
```