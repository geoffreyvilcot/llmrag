import json

class Config(object):
    def __init__(self, conf_file="config.json"):
        with open(conf_file, "rt", encoding="utf8") as f :
            jconf = json.load(f)

        self.model_path = jconf['model_path']
        self.ingest_files_dir = jconf['ingest_files_dir']
        self.vector_db_file = jconf['vector_db_file']
