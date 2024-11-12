# Flask REST API example
from flask import Flask, json, render_template
from graphs import load_tokens, load_activations, visualize_feature
from graphs import get_statistics

import networkx as nx

import json

tokens = None
tokenizer = None
activations = None
locations = None

def create_app():
    global tokens, tokenizer, activations, locations
    app = Flask(__name__)
    print("loading tokens")
    tokens, tokenizer = load_tokens("google/gemma-2-9B", "kh4dien/fineweb-100m-sample")
    activations, locations = load_activations('features/11_0_3275.safetensors')
    return app

app = create_app()

# Define dynamic routes that will visualize a specific feature
@app.route('/features/<int:n>', methods=['GET'])
def get_graph(n):
    graphs, contexts, activation_dicts = visualize_feature(n, activations, locations, tokens, tokenizer)
    statistics = get_statistics(n, activations, locations, tokens, tokenizer)

    return render_template("index.html", 
                           graphs=graphs, 
                           contexts=contexts, 
                           statistics=statistics,
                           activation_dicts=activation_dicts)

@app.route('/')
def index():    
    return 'Hello, welcome to my Flask app!'



    
