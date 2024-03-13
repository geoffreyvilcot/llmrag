from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import numpy as np

class My_embedding(object) :
    def __init__(self, model_name = "distilbert-base-nli-mean-tokens"):
        #"distilbert-base-nli-mean-tokens"
        self.model = SentenceTransformer(model_name)
        self.train_examples = []
        pass

    def add_train_example(self, texts : [str], label : float):
        self.train_examples.append(InputExample(texts=texts, label=label))

    def fit(self, epochs, batch_size, warmup_steps=100):
        # Define your train dataset, the dataloader and the train loss
        train_dataloader = DataLoader(self.train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(self.model)

        # Tune the model
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=warmup_steps)

    def encode(self, sentences):
        return self.model.encode(sentences)
    def save(self, filename):
        self.model.save(filename)


if __name__ == '__main__':

    # Define the model. Either from scratch of by loading a pre-trained model


    # Define your train examples. You need more than just two examples...
    train_examples = [
        InputExample(texts=["My first sentence", "My second sentence"], label=1.),
        InputExample(texts=["Another pair", "Unrelated sentence"], label=2.),
    ]

    model = My_embedding("path_finder_emb.pk")


    # Our sentences to encode
    sentences = [
        """**École** Enchantement [émotion, effet mental] ; **Niveau** Bard 1, Hyp 1,
Ens/Mag 2, Psy 2, Sor 2  
**Temps d’incantation** 1 action simple  
**Composantes** V, G  
**Portée** courte (7,50 m + 1,50 m/2 niveaux) / (5 c \+ 1 c/2 niveaux)  
**Cible** une créature vivante  
**Durée** 1 minute/niveau  
**Jet de sauvegarde** Volonté pour annuler (inoffensif) ; **Résistance à la
magie** oui  """,
        """La cible du sort devient un objet d’adoration pour ceux qu’elle tente
d’affecter avec un test de Diplomatie ou de combat de spectacle. Si la cible
n’est pas impliquée dans un combat, elle reçoit un bonus de moral de +2 aux
tests de Diplomatie pour influencer une créature. Si elle est engagée dans un
combat de spectacle , elle gagne un bonus de moral de +2 aux tests de combat
de spectacle.""",
        "**École** Évocation [feu, dépendant du langage]; **Niveau** Alch 2, Bard 2,",
        "Le personnage débite une tirade insultante si vicieuse et méprisante que tous",
        "En analysant des données et en fournissant des informations, l’IA générative renforce les processus de prise de décision"
    ]

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    # Print the embeddings
    # for sentence, embedding in zip(sentences, embeddings):
    #     print("Sentence:", sentence)
    #     print("Embedding:", embedding)
    #     print("")

    for i in range(len(embeddings)) :
        for j in range(i, len(embeddings)) :
            distance = np.linalg.norm(embeddings[i] - embeddings[j])
            print(f"{sentences[i]}\n--\n{sentences[j]}\n=> {distance}\n=======================\n")