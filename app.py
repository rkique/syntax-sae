# Flask REST API example
from flask import Flask, json, render_template
from graphs import load_tokens, load_activations, visualize_feature, visualize_feature_flat
from graphs import merged_parse_trees, get_total_occurrences
from graphs import get_joint_parse_tree, jsonify, context_to_dict
from graphs import get_statistics
import sqlite3
import networkx as nx
import json
import numpy as np
from huggingface_hub import login


tokens = None
tokenizer = None
activations = None
locations = None

def create_app():
    global tokens, tokenizer, activations, locations
    app = Flask(__name__)
    print("loading tokens")
    # tokens, tokenizer = load_tokens("google/gemma-2-9B", "kh4dien/fineweb-100m-sample")
    # activations, locations = load_activations('features/11_0_3275.safetensors')
    return app

app = create_app()

def get_graph_from_db(n):
    conn = sqlite3.connect("feature_graphs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT graphs, contexts, activation_dicts FROM features WHERE id = ?", (n,))
    result = cursor.fetchone()
    conn.close()

    if result:
        graphs = json.loads(result[0])
        contexts = json.loads(result[1])
        activation_dicts = json.loads(result[2])
        return graphs, contexts, activation_dicts
    else:
        return [], [], []

@app.route('/joint/<int:n>', methods=['GET'])
def get_joint(n):
    graphs, contexts, activation_dicts = get_graph_from_db(n)
    graphs = [json.loads(graph) for graph in graphs]
    graphs = get_joint_parse_tree(graphs, is_merge=False)
    graph = json.dumps(graphs)
    return render_template("feature-joint.html", 
                        n=n,
                        graphs=[graph], 
                        contexts=[], 
                        statistics=[],
                        activation_dicts=[])

@app.route('/cytoscape', methods=['GET'])
def get_cytoscape():
    return render_template("cytoscape.html")

@app.route('/merged/<int:n>', methods=['GET'])
def get_merged(n):
    graphs, contexts, activation_dicts = get_graph_from_db(n)
    #graphs = [json.loads(graph) for graph in graphs]
    graphs = [context_to_dict(context, activation_dict) for context, activation_dict in zip(contexts, activation_dicts)]
    graphs = merged_parse_trees(graphs)
    # merged = merged_parse_trees(graphs) #merged is list of dicts
    # depths = [get_total_occurrences(tree) for tree in merged]
    # merged = sorted(zip(merged, depths), key=lambda x:x[1], reverse=True)
    # merged = [x[0] for x in merged]
    graph = json.dumps(get_joint_parse_tree(graphs, is_merge=True))
    return render_template("feature-merge.html", 
                        n=n,
                        graphs=[graph], 
                        contexts=[], 
                        statistics=[],
                        activation_dicts=[])

# Define dynamic routes that will visualize a specific feature
@app.route('/features/<int:n>', methods=['GET'])
def get_graph(n):
    graphs, contexts, activation_dicts = get_graph_from_db(n)
    #statistics = get_statistics(n, activations, locations, tokens, tokenizer)
    statistics = {
        'num_activations': 0,
        'pos_pcts': {}}
    
    return render_template("feature-context.html", 
                           n=n,
                           graphs=graphs, 
                           contexts=contexts, 
                           statistics=statistics,
                           activation_dicts=activation_dicts)

@app.route('/active/<int:n>', methods=['GET'])
def get_activations(n):
    context_acts = visualize_feature_flat(n, activations, locations, tokens, tokenizer)
    print(type(context_acts))
    return render_template("active.html",    
                           contexts=context_acts)

@app.route('/')
def index():    
    points = np.load('pos_map.npy')
    # Prepare data for visualization
    scatter_data = [
        {
            "x": float(points[i][1]), #transpose.
            "y": float(points[i][0]),
            "feature": i  # Create a link for each feature
        }
        for i in range(points.shape[0])
    ]
    return render_template("index.html", scatter_data=scatter_data)

app.run(host='0.0.0.0', port=8080)
