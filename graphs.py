
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
from itertools import accumulate

TOTAL_BATCHES = 39039
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")

Token.set_extension("custom_tag", default=0, force=True)

def get_context(batch: list[str], pos: int, n=5) -> list[str]:
    context = [s for s in batch[pos-n:pos+n] if s != '\n']
    print(f'context is {context}')
    return context

def view_batch(n: int, tokens: torch.Tensor, tokenizer) -> list[str]:
    """
    Returns the nth batch from the given tokens tensor as a list of strings using the given tokenizer.
    
    Args:
        n (int): Index of the batch to retrieve.
        tokens (torch.Tensor): Tokens tensor to retrieve from.
        tokenizer (transformer_lens.utils.LanguageModel.tokenizer): Tokenizer to use for decoding.
    
    Returns:
        list[str]: List of strings in the nth batch.
    """
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

def position_to_char_indice(context: list[str], position: int) -> int:
    char_indices = list(accumulate(len(s) for s in context))
    character_position = char_indices[position]
    return character_position


def positions_to_char_indices(context: list[str], token_positions: list[int]) -> list[int]:
    # Concatenate all context strings to form a single string
    char_indices = list(accumulate(len(s) for s in context))
    character_positions = [char_indices[position] for position in token_positions]
    return character_positions

context = ["I", "like", "rock", "climbing"]
positions = [0, 2]
char_pos = positions_to_char_indices(context, positions)
assert(char_pos == [1,9])

#Context is the sentence broken into tokens without punctuation marks and with spaces preserved. Doc is the sentence spacy Doc.
def make_parse_tree(context: list[str], doc : spacy.tokens.Doc, positions: list[int], activations: list[float]) -> spacy.tokens.Token:
    positions = [position - 1 for position in positions]
    a = [context[position] for position in positions]
    #print(f'[MAKE_PARSE] {context=} a are {a} ')
    character_positions = positions_to_char_indices(context, positions)
    #Set character indices at various activations to value.
    for pos, act in zip(character_positions, activations):
        activation_node = None
        for token in doc:
            if (int(token.idx) <= pos <= int(token.idx + len(token))):
                activation_node = token
        activation_node._.custom_tag = act
    # Find the root node (the one with dep_ == "ROOT")
    root_node = next(node for node in list(doc)[::-1] if node.dep_ == "ROOT")
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
        "tag": float(token._.custom_tag),
        "children": [node_to_dict(child) for child in token.children]  # Recursively add children
    }

def jsonify(root_token):
    """Converts a spaCy root node and its entire dependency tree to a JSON string."""
    root_dict = node_to_dict(root_token)
    return json.dumps(root_dict, indent=4)

def ttnp(tensor):
    return tensor.detach().cpu().item()

def batch_dicts(n, activations, locations, k=5):
    idx = locations[:,2]== n 
    locations = locations[idx]
    activations = activations[idx]
    total_batches = TOTAL_BATCHES
    batch_dicts = [{'i': i, 'positions': [], 'activations': []} for i in range(0, total_batches + 1)]
    for location, activation in zip(locations, activations):
        d = batch_dicts[location[0]]
        d['positions'].append(int(ttnp(location[1])))
        d['activations'].append(activation)
    
    batchdict_tuple = [(d, max(d['activations']) if d['activations'] else float('-inf')) for d in batch_dicts]
    sorted_tuples = sorted(batchdict_tuple, key=lambda x: x[1], reverse=True)[:5]
    top_n_dicts = [d for d, _ in sorted_tuples[:n]]
    return top_n_dicts

def get_sentence_at_index(doc, char_index):
    for sent in doc.sents:
        if sent.start_char <= char_index < sent.end_char:
            sent_tokens = [token.text for token in sent]
            return sent_tokens, sent, sent.start_char, sent.end_char
    raise Exception("No sentence at index")

def get_token_idx(string_list: list[str], char_index: int):
    # Concatenate the list of strings into a single string
    current_length = 0
    for i, s in enumerate(string_list):
        current_length += len(s)
        if current_length >= char_index:
            return i  
    raise Exception(f"Char index {char_index} not found in List: length {current_length}")

context = ["I", "like", "rock", "climbing"]
assert(get_token_idx(context, 4) == 1)
assert(get_token_idx(context, 8) == 2)

# Return active percen
def visualize_feature(n, activations, 
                      locations, tokens, tokenizer, k=5) -> tuple[list[str], list[str], list[dict]]:
    print(f"Visualizing Feature {n}")
    idx = locations[:,2]== n
    locations = locations[idx]
    activations = activations[idx]
    top_dicts = batch_dicts(n, activations, locations, 5)
    #top_dicts contain pos, act lists.
    parse_trees = []
    contexts = []
    activation_dicts = []
    for d in top_dicts:
        positions = d['positions']
        activations = d['activations']

        #get token position around highest activation
        batch = view_batch(int(d['i']), tokens, tokenizer)
        max_position_index = activations.index(max(activations))
        max_token_idx = positions[max_position_index]
        
        #convert token position to character position
        doc = nlp("".join(batch))
        #print(f'BATCH: {batch=}')
        max_char_idx = position_to_char_indice(batch, max_token_idx)
        #print(f'act. token_index: {max_token_idx} act. char index: {max_char_idx} in {doc}')
        context_list, sent, start_char, end_char = get_sentence_at_index(doc, max_char_idx)
        sent = sent.as_doc()
        print(f'{context_list=}')
        start_idx = get_token_idx(batch, start_char)
        end_idx = get_token_idx(batch, end_char)
        #print(f'{batch=} {start_idx=} {end_idx=}')
        def offset(position):
            return position - start_idx
        
        filtered_positions, filtered_activations = zip(*[(offset(position), activation) 
                    for (position, activation) in zip(positions, activations)
                    if 0 <= (offset(position)) < len(context_list)]) #this makes some negative.
        
        activation_dict = {filtered_positions[i]: float(filtered_activations[i])
                           for i in range(len(filtered_positions))}
        
        #print(f'f{context_list=} {filtered_positions=}')
        print(f'[START END] {start_idx=} {end_idx=}')
        token_context = batch[start_idx:end_idx]
        parse_tree = make_parse_tree(token_context, sent,
                                     filtered_positions, filtered_activations)
        
        parse_trees.append(jsonify(parse_tree))
        contexts.append(batch[start_idx:end_idx])
        activation_dicts.append(activation_dict)

    return parse_trees, contexts, activation_dicts
