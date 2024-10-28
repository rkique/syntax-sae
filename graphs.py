
from nnsight import LanguageModel
from datasets import load_dataset
import transformer_lens.utils as utils
import spacy
import networkx as nx
from safetensors.numpy import load_file
import numpy as np
import torch

nlp = spacy.load("en_core_web_sm")

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

#TODO: aggregate indices for parse trees
def make_parse_tree(context: list[str], act_idx: int) -> list[list[dict]]:
    """
    Takes a list of strings, and returns a list of dictionaries representing a parse tree for the input.
    The dictionaries contain the following keys:
        - "text": the text of the token
        - "lemma": the lemma of the token
        - "pos": the part of speech of the token
        - "dependency": the dependency label of the token
        - "children": a list of strings representing the children of the token
    """
    doc = nlp(context)
    parse_tree = []
    for token in doc:
        node = {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dependency": token.dep_,
            "children": [child.text for child in token.children]
        }
        parse_tree.append(node)
    return parse_tree

## Define joint parse tree from individual parse trees.
def joint_parse_tree(contexts, act_idxs):
    #we want to aggregate shared POS in preceding and forward contexts.
    #1. We aggregate context trees no matter what.
    #2. We combine shared POS while retaining the tokens
    for context, act_idx in zip(contexts, act_idxs):
        parse_tree = make_parse_tree(context, act_idx)
    #TODO
    return None


def make_graph(parse_dict : list[dict]):
    G = nx.DiGraph()
    for token in parse_dict:
        token_id = f'{token['text']} [{token['pos']}]'
        G.add_node(token_id)
        for child in token['children']:
            G.add_edge(token_id, child)
    pos = nx.spring_layout(G)  # You can also try 'shell_layout', 'circular_layout', etc.
    return G, pos

def visualize_feature(n, activations, locations, tokens, tokenizer, k=5):
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

    #visualize just the top tree.
    for d in sorted_location_dicts[0:1]:
        batch = view_batch(int(d['batch']), tokens, tokenizer)
        if batch != None:
            pos = d['position']
            n = 5
            context = get_context(batch, pos, n=n) #todo: figure out better parse tree context e.g. by punct.
            parse_tree = make_parse_tree(context, n)
            G, pos = make_graph(parse_tree)
            return G, pos
        