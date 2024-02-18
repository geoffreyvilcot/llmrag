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

