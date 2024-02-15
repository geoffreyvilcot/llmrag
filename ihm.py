import random

import wx

import copy
import time
import sys
from llama_cpp import Llama
from config import Config
import pickle
import numpy as np
from config import Config
import json
import time

from threading import Thread

import getopt

class Thread_Worker(Thread) :
    def __init__(self, text_ctrl : wx.TextCtrl, text_debug : wx.TextCtrl, button : wx.Button, iterations : int, temperature : float):
        Thread.__init__(self)
        self.text_ctrl = text_ctrl
        self.text_debug = text_debug
        self.iterations = iterations
        self.temperature = temperature
        self.button = button
        self.force_stop = False
    def run(self) -> None:
        text = self.text_ctrl.GetValue()

        question_embeddings = np.array([llm.embed(text)])
        D, I = index.search(question_embeddings, k=2)  # distance, index
        retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

        prompt = f"""
        Le contexte est décrit ci-dessous.
        ---------------------
        {retrieved_chunk}
        ---------------------
        A partir de ces informations de contexte et avec aucun savoir préalable, Répondre à la question.
        Question: {text}
        Reponse:
        """

        print(prompt)

        start_t = time.time()
        output = llm(
            prompt,  # Prompt
            max_tokens=self.iterations,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["Query:", "Question:"],  # Stop generating just before the model would generate a new question
            echo=False  # Echo the prompt back in the output
        )  # Generate a completion, can also call create_completion
        end_t = time.time()
        print(output)
        self.text_ctrl.AppendText("\n\n" + output['choices'][0]['text'])

        prompt_tokens_per_sec = int(output['usage']['prompt_tokens']) / (end_t-start_t)
        completion_tokens_per_sec = int(output['usage']['completion_tokens']) / (end_t-start_t)

        self.text_debug.Clear()
        self.text_debug.WriteText(f"Prompt: {prompt_tokens_per_sec:0.2f} tokens/s | Completion: {completion_tokens_per_sec:0.2f} tokens/s")
        self.button.Enable()
        # print(output_str)
        print('*'*40)


class MyFrame(wx.Frame):
    def __init__(self, parent, title, initial_text=""):
        wx.Frame.__init__(self, parent, title=title, size=(800, 600))

        self.th = None
        # Créer un panneau
        panel = wx.Panel(self)

        # Créer une zone de saisie
        self.text_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
        self.text_ctrl.WriteText(initial_text)

        self.len_ctrl = wx.TextCtrl(panel, size=(100,-1))
        self.len_ctrl.WriteText("500")

        self.temp_ctrl = wx.TextCtrl(panel, size=(100,-1))
        self.temp_ctrl.WriteText("1")

        self.debugtext = wx.TextCtrl(panel, size=(100,-1), style=wx.TE_READONLY)



        # self.text_ctrl.SetStyle(wx.TE_MULTILINE)

        # Créer un bouton
        button = wx.Button(panel, label="OK")
        # Associer une fonction à l'événement du bouton
        button.Bind(wx.EVT_BUTTON, self.on_button)

        # Créer un bouton
        button_stop = wx.Button(panel, label="Stop")
        # Associer une fonction à l'événement du bouton
        button_stop.Bind(wx.EVT_BUTTON, self.on_button_stop)

        # Créer un sizer pour positionner les widgets

        sizerH = wx.BoxSizer(wx.HORIZONTAL)
        sizerH.Add(self.len_ctrl, 0,  wx.EXPAND | wx.ALL, 5)
        sizerH.Add(self.temp_ctrl, 0,  wx.EXPAND | wx.ALL, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(sizerH, 0,  wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.text_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.debugtext, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(button, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(button_stop, 0, wx.EXPAND | wx.ALL, 5)

        # Affecter le sizer au panneau
        panel.SetSizer(sizer)
        self.panel = panel
        self.button = button

    def on_button(self, event):
        iteration=int(self.len_ctrl.GetValue())
        temperature = float(self.temp_ctrl.GetValue())

        self.th = Thread_Worker(self.text_ctrl, self.debugtext, self.button, iteration, temperature)
        self.th.start()
        self.button.Disable()
        # Récupérer le texte saisi
        # Afficher le texte dans la console
        # print(text)
    def on_button_stop(self, event):
        if self.th != None :
            self.th.force_stop = True



if __name__ == "__main__":
    conf_file_name = "config.json"

    opts, args = getopt.getopt(sys.argv[1:],"hc:")
    for opt, arg in opts:
        if opt == '-h':
            print(sys.argv[0] + ' -c <conf_file>')
            sys.exit()
        elif opt in ("-c"):
            conf_file_name = arg

    conf = Config(conf_file=conf_file_name)

    llm = Llama(
        model_path=conf.model_path,
        embedding=True,
        n_gpu_layers=conf.n_gpu_layers, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        n_ctx=conf.n_ctx,  # Uncomment to increase the context window
    )
    with open(conf.vector_db_file, 'rb') as file:
        index, chunks = pickle.load(file)

    # Créer une application
    app = wx.App()

    # Créer une fenêtre
    frame = MyFrame(None, "Llama rag")

    # Afficher la fenêtre
    frame.Show()

    # Lancer l'application
    app.MainLoop()