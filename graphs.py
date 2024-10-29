
from nnsight import LanguageModel
from datasets import load_dataset
import transformer_lens.utils as utils
import spacy
import networkx as nx
from safetensors.numpy import load_file
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import json
from spacy.tokens import Token
nlp = spacy.load("en_core_web_sm")
Token.set_extension("custom_tag", default=0)

def get_context(batch: list[str], pos: int, n=5) -> str:
    
    context = ''.join([s for s in batch[pos-n:pos+n] if s != '\n'])
    return context

def view_batch(n: int, tokens: torch.Tensor, tokenizer) -> list[str]:
    assert(tokens.ndim == 2 and tokens.shape[1] == 256)
    if n >= len(tokens):
        return None
    doc = tokens[n]
    return tokenizer.batch_decode(doc)

def load_tokens(tokenizer_name, dataset_name):
    """
    Loads tokens from finetuned gemma model and shuffles them.
    Returns:
        tokens (list of lists of int): list of tokenized sentences where each sentence is a list of int.
    """
    tokenizer = LanguageModel(tokenizer_name).tokenizer
    data = load_dataset(dataset_name, name="", split="train")
    tokens = utils.tokenize_and_concatenate(data, tokenizer, max_length=256,column_name="text")
    tokens = tokens.shuffle(22)["tokens"]
    return tokens, tokenizer

def load_activations(path):
    feature_dict = load_file(path)
    activations,locations = feature_dict['activations'], feature_dict['locations']
    locations = torch.tensor(locations.astype(np.int64))
    return activations, locations

class Node:
    def __init__(self, text, lemma, pos, dependency, children):
        self.text = text
        self.lemma = lemma
        self.pos = pos
        self.dependency = dependency
        self.children = children

    def __repr__(self):
        return f"{self.text} [{self.pos}]"
    
#TODO: aggregate indices for parse trees
def make_parse_tree(context: list[str], act_idx):
    doc = nlp(context)
    activation_node = doc[act_idx]
    activation_node._.custom_tag = 1
    # Find the root node (the one with dep_ == "ROOT")
    root_node = next(node for node in doc if node.dep_ == "ROOT")
    return root_node

## Define joint parse tree from individual parse trees.
def joint_parse_tree(contexts, act_idxs):
    #we want to aggregate shared POS in preceding and forward contexts.
    #1. We aggregate context trees no matter what.
    #2. We combine shared POS while retaining the tokens
    for context, act_idx in zip(contexts, act_idxs):
        parse_tree = make_parse_tree(context, act_idx)
    #TODO
    return None

def node_to_dict(token):
    return {
        "text": token.text,
        "lemma": token.lemma_,
        "pos": token.pos_,
        "dep": token.dep_,
        "tag": token._.custom_tag,
        "children": [node_to_dict(child) for child in token.children]  # Recursively add children
    }

def jsonify(root_token):
    """Converts a spaCy root node and its entire dependency tree to a JSON string."""
    root_dict = node_to_dict(root_token)
    return json.dumps(root_dict, indent=4)

def visualize_feature(n, activations, 
                      locations, tokens, tokenizer, k=5) -> list[str]:
    """
    top k activated contexts for a feature.
    """
    print(f"Visualizing Feature {n}")
    idx = locations[:,2]== n
    locations = locations[idx]
    activations = activations[idx]
    location_dicts = []

    for location, activation in zip(locations, activations):
        d = {}
        d['batch'] = location[0]
        d['position'] = location[1]
        d['feature'] = location[2]
        d['activation'] = activation
        location_dicts.append(d)
    sorted_location_dicts = sorted(location_dicts, key=lambda x: x['activation'], reverse=True)
    
    #return just the top tree
    parse_trees = []
    for d in sorted_location_dicts[0:5]:
        batch = view_batch(int(d['batch']), tokens, tokenizer)
        if batch != None:
            pos = d['position']
            n = 5
            context = get_context(batch, pos, n=n) #todo: figure out better parse tree context e.g. by punct.
            parse_tree = make_parse_tree(context, n)
            parse_trees.append(jsonify(parse_tree))
    return parse_trees