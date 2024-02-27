import faiss
import pickle
from config import Config
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct

import os

class Vector_DB :
    def __init__(self, conf : Config, dim):
        pass

    def reset(self):
        pass

    def add(self, vector, payload):
        pass

    def search(self, vector, k=3):
        pass

    def save(self):
        pass
    def load(self):
        pass

class Vector_DB_Faiss(Vector_DB) :
    def __init__(self, conf : Config, dim=None):
        super().__init__(conf, dim)
        self.conf = conf
        if dim is None :
            self.index = None
        else :
            self.index = faiss.IndexFlatL2(dim)
        self.payloads = []
        self.k_vector = 3
    def add(self, vector, payload):
        self.index.add(vector)
        for p in payload :
            self.payloads.append(p)
        pass

    def search(self, vector, k=3):
        D, I = self.index.search(vector, k=k)  # distance, index

        retrieved_payloads = [self.payloads[i] for i in I.tolist()[0]]
        return retrieved_payloads

    def save(self):
        with open(self.conf.vector_db_file, 'wb') as file:
            pickle.dump((self.index, self.payloads), file)
    def load(self):
        with open(self.conf.vector_db_file, 'rb') as file:
            self.index, self.payloads = pickle.load(file)




class Vector_DB_Qdrant(Vector_DB) :
    def __init__(self, conf : Config, dim=None):
        super().__init__(conf, dim)
        self.conf = conf
        if self.conf.qdrant_local :
            self.client = QdrantClient(path=self.conf.vector_db_file)
        else :
            self.client = QdrantClient(self.conf.qdrant_host, port=self.conf.qdrant_port)
        self.k_vector = 3
        self.dim = dim

    def reset(self):
        self.client.recreate_collection(
            collection_name=self.conf.qdrant_collection,
            vectors_config=VectorParams(size= self.dim, distance=Distance.DOT),
        )

    def add(self, vector, payload):
        vector_points = []
        for i in range(vector.shape[0]) :
            vector_points.append(PointStruct(id=i, vector=vector[i].tolist(), payload={"data": payload[i]}))
            if len(vector_points) > 100 :
                operation_info = self.client.upsert(
                    collection_name=self.conf.qdrant_collection,
                    wait=True,
                    points=vector_points,
                )
                vector_points = []
                print(operation_info)

        if len(vector_points) > 0 :
            operation_info = self.client.upsert(
                collection_name=self.conf.qdrant_collection,
                wait=True,
                points=vector_points,
            )
            vector_points = []
            print(operation_info)

    def search(self, vector, k=3):
        search_result = self.client.search(
            collection_name=self.conf.qdrant_collection, query_vector=vector[0].tolist(), limit=k
        )
        # print(search_result)
        retrieve_payloads = [ result.payload['data'] for result in search_result]
        return retrieve_payloads
        # D, I = self.index.search(vector, k=self.k_vector)  # distance, index
        #
        # retrieved_payloads = [self.payloads[i] for i in I.tolist()[0]]
        # return retrieved_payloads


