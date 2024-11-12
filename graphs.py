
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
from collections import defaultdict, Counter


TOTAL_BATCHES = 39039
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")

Token.set_extension("custom_tag", default=0, force=True)

def get_context(loc, k: int, tokens: torch.Tensor, tokenizer) -> list[str]:
    doc = tokens[loc[0]]
    context = doc[loc[1]-k:loc[1]+k]
    return tokenizer.batch_decode(context)

def get_batch_text(n: int, tokens: torch.Tensor, tokenizer) -> list[str]:
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
    char_indices = [0] + list(accumulate(len(s) for s in context))
    character_positions = [char_indices[position] for position in token_positions]
    return character_positions

context = ["I", "like", "rock", "climbing"]
positions = [0, 2]
char_pos = positions_to_char_indices(context, positions)
assert(char_pos == [0,5])

#Context is the sentence broken into tokens without punctuation marks and with spaces preserved. Doc is the sentence spacy Doc.
def make_parse_tree(context: list[str], doc : spacy.tokens.Doc, positions: list[int], activations: list[float]) -> spacy.tokens.Token:
    #print(f'[MAKE_PARSE] positions are {positions}')
    #print(f'[MAKE_PARSE] {context=} a are {a} ')
    character_positions = positions_to_char_indices(context, positions)
    #Set character indices at various activations to value.
    for pos, act in zip(character_positions, activations):
        activation_node = None
        for token in doc:
            #print(f'{token.idx} {pos} {token.idx+len(token)} {token.text=} ')
            if (int(token.idx) <= pos <= int(token.idx + len(token))):
                activation_node = token
        activation_node._.custom_tag = act
    # Find the root node (the one with dep_ == "ROOT")
    root_node = next(node for node in list(doc)[::-1] if node.dep_ == "ROOT")
    return root_node

## Define joint parse tree from individual parse trees.
def joint_parse_tree(contexts: list[list[str]], 
                     docs: list[spacy.tokens.Doc], 
                     positions: list[list[int]],
                     activations: list[list[float]]):
    # define a joint parse tree from here...
    # how to isolate parse trees.
    for context, doc, pos, act in zip(contexts, docs, positions, activations):
        parse_tree = make_parse_tree(context, doc, pos, act)

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

#Returns position-activation dicts by number of features.
def batch_dicts(n, activations, locations, k=12):
    total_batches = TOTAL_BATCHES
    batch_dicts = [{'i': i, 'positions': [], 'activations': []} for i in range(0, total_batches + 1)]
    for location, activation in zip(locations, activations):
        d = batch_dicts[location[0]]
        d['positions'].append(int(ttnp(location[1]))) #location[0] is batch, location[1] is activation
        d['activations'].append(activation)
    
    #sort by max activation
    batchdict_tuple = [(d, max(d['activations']) if d['activations'] else float('-inf')) for d in batch_dicts]
    sorted_tuples = sorted(batchdict_tuple, key=lambda x: x[1], reverse=True)[:k]
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
    return None

context = ["I", "like", "rock", "climbing"]
assert(get_token_idx(context, 4) == 1)
assert(get_token_idx(context, 8) == 2)

# Return the part of speech most active.
# Return the number of contiguous tokens / scale metrics.
def get_statistics(n, activations, locations, tokens, tokenizer) -> dict:
    idx = locations[:,2]== n
    locations = locations[idx]
    activations = activations[idx]
    num_activations = len(activations)
    avg_act = np.mean(activations)
    pos_pcts = {}
    #we want all activations and positions above threshold. 
    filtered_locations, filtered_activations = zip(*[(location, activation) 
                    for (location, activation) in zip(locations, activations)
                    if avg_act <= activation])
    
    pos_pcts = defaultdict()
    #For each location, add the activation to the corresponding part of speech category
    for loc, act in zip(filtered_locations, filtered_activations):
        WINDOW_SIZE = 3
        batch = get_context(loc, WINDOW_SIZE, tokens, tokenizer)
        #print(f'{batch=}')
        if len(batch) == 2 * WINDOW_SIZE:
            char_idx = position_to_char_indice(batch, WINDOW_SIZE) #token index to char index
            doc = nlp("".join(batch))
            active_token = next((token for token in doc 
                                if token.idx <= char_idx < token.idx + len(token)), None)
            if active_token:
                pos = active_token.pos_
                omit_pos = ['PUNCT', 'SYM', 'X', 'EOL', 'SPACE']
                if pos not in omit_pos:
                    pos_pcts[pos] = pos_pcts.get(pos, 0) + act
    act_sum = sum(pos_pcts.values())
    #Create a dictionary with the percentage of activations for each part of speech
    pos_pcts = {k: float(v / act_sum) for k, v in pos_pcts.items()}
    statistics = {
        'num_activations': num_activations
        , 'pos_pcts': pos_pcts
    }
    return statistics

CONTEXTS = 100

#draws a flatter tree from the top activating contexts.
def get_context_activations(n, activations, locations, tokens, tokenizer, ws=30) -> tuple[list[str], list[dict]]:
    print(f"Visualizing Feature {n}")
    idx = locations[:,2]== n
    locations = locations[idx]
    activations = activations[idx]
    top_dicts = batch_dicts(n, activations, locations, CONTEXTS)
    #top_dicts contain pos, act lists.
    context_activations = []
    for d in top_dicts:
        positions = d['positions']
        activations = d['activations']
        #access batch text and pipeline through spacy
        batch = get_batch_text(int(d['i']), tokens, tokenizer)
        doc = nlp("".join(batch))
        #get token position around highest activation
        max_position_index = activations.index(max(activations))
        max_token_idx = positions[max_position_index]
        max_char_idx = position_to_char_indice(batch, max_token_idx)
        #this sentence bound is arbitrary.
        start_char = max_char_idx - ws #char
        end_char = max_char_idx + ws
        start_idx = get_token_idx(batch, start_char) if start_char >= 0 else 0
        end_idx = get_token_idx(batch, end_char) if end_char < len(batch) else len(batch)
        token_context = batch[start_idx:end_idx]
        def offset(position):
            return position - start_idx
        #print(f'{token_context=} {max_char_idx=} {start_char=} {end_char=}')  
        filtered_positions, filtered_activations = zip(*[(offset(position), activation) 
                    for (position, activation) in zip(positions, activations)
                    if 0 <= (offset(position)) <= ws * 2]) #filters batch activations to window
        
       #list[(token, activation)]
        context_activation = []
        for i in range(0, len(token_context)):
            #if there is a corresponding activation, index it.
            if i in filtered_positions:
                activation = float(filtered_activations[filtered_positions.index(i)])
            else:
                activation = 0
            context_activation.append((token_context[i], activation))
        context_activations.append(context_activation)
    #list[list[(token, activation)]]
    return context_activations

def visualize_feature_flat(n, activations, locations, tokens, tokenizer, ws=30):
    context_acts = get_context_activations(n, activations, locations, tokens, tokenizer, ws=ws)
    contexts_around_max = []
    view_ws = 2
    for context in context_acts:
        #get index of max of token activation on context
        highest_index, _ = max(enumerate(context), key=lambda x: x[1][1])
        context_around_max = []
        for offset in range(-view_ws, view_ws + 1): 
            position = highest_index + offset
            if 0 <= position < len(context): 
                token, activation = context[position]
                token = token.strip()
                activation = float(activation)
                context_around_max.append((token, activation))
        contexts_around_max.append(context_around_max)
    #get the most common activations from the counters.
    #log the activations.
    # for c in c_array:
    #     most_common_tokens = [(token, np.log(total_act)) for token, total_act in c.most_common(3)]
    #     top_tokens_around_max.append(most_common_tokens)

    return contexts_around_max
    return context_acts



#Given activations over contexts, returns activations
def visualize_feature(n, activations, 
                      locations, tokens, tokenizer, k=5) -> tuple[list[str], list[str], list[dict]]:
    print(f"Visualizing Feature {n}")
    idx = locations[:,2]== n
    locations = locations[idx]
    activations = activations[idx]
    top_dicts = batch_dicts(n, activations, locations, 12)
    
    #top_dicts contain pos, act lists.
    parse_trees = []
    contexts = []
    activation_dicts = []
    for d in top_dicts:
        positions = d['positions']
        activations = d['activations']

        #access batch text and pipeline through spacy
        batch = get_batch_text(int(d['i']), tokens, tokenizer)
        doc = nlp("".join(batch))

        #get token position around highest activation
        max_position_index = activations.index(max(activations))
        max_token_idx = positions[max_position_index]

        max_char_idx = position_to_char_indice(batch, max_token_idx)
        context_list, sent, start_char, end_char = get_sentence_at_index(doc, max_char_idx)
        sent = sent.as_doc()
        #print(f'{context_list=}')
        start_idx = get_token_idx(batch, start_char)
        end_idx = get_token_idx(batch, end_char)
        #print(f'{batch=} {start_idx=} {end_idx=}')
        def offset(position):
            return position - start_idx
        
        filtered_positions, filtered_activations = zip(*[(offset(position), activation) 
                    for (position, activation) in zip(positions, activations)
                    if 0 <= (offset(position)) < len(context_list)]) 
        
        activation_dict = {filtered_positions[i]: float(filtered_activations[i])
                           for i in range(len(filtered_positions))}
        
        #print(f'f{context_list=} {filtered_positions=}')
        #print(f'[START END] {start_idx=} {end_idx=}')
        token_context = batch[start_idx:end_idx]
        parse_tree = make_parse_tree(token_context, sent,
                                     filtered_positions, filtered_activations)
        
        parse_trees.append(jsonify(parse_tree))
        contexts.append(batch[start_idx:end_idx])
        activation_dicts.append(activation_dict)

    return parse_trees, contexts, activation_dicts
