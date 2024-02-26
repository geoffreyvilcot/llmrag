import json

class Config(object):
    def __init__(self, conf_file="config.json"):
        with open(conf_file, "rt", encoding="utf8") as f :
            jconf = json.load(f)

        self.model_path = jconf['model_path']
        self.ingest_files_dir = jconf['ingest_files_dir']
        self.vector_db_file = jconf['vector_db_file']
        self.n_ctx = int(jconf['n_ctx'])
        self.n_gpu_layers = int(jconf['n_gpu_layers'])

        self.prompt_template = jconf['prompt_template']

        self.use_qdrant = False
        self.qdrant_host = ""
        self.qdrant_port = ""
        self.qdrant_collection = ""
        if "use_qdrant" in jconf and bool(jconf['use_qdrant']) :
            self.use_qdrant = True
        if "qdrant_host" in jconf :
            self.qdrant_host = jconf['qdrant_host']
        if "qdrant_port" in jconf :
            self.qdrant_port = jconf['qdrant_port']
        if "qdrant_collection" in jconf :
            self.qdrant_collection = jconf['qdrant_collection']

        self.use_rag = False
        if "use_rag" in jconf and bool(jconf['use_rag']) :
            self.use_rag = True

        self.chunks_mode = jconf['chunks_mode']

        if "listen_bind" in jconf :
            self.listen_bind = jconf['listen_bind']
        else :
            self.listen_bind = "127.0.0.1"

        if "listen_port" in jconf:
            self.listen_port = int(jconf['listen_port'])
        else:
            self.listen_port = 49283

        if "ingest_max_tokens" in jconf:
            self.ingest_max_tokens = int(jconf['ingest_max_tokens'])
        else:
            self.ingest_max_tokens = 256


