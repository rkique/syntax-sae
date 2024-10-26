# Flask REST API example
from flask import Flask, jsonify, render_template

from graphs import load_tokens, load_activations, visualize_feature
global tokens, tokenizer, activations, locations
import networkx as nx

app = Flask(__name__)

# Define dynamic routes that will visualize a specific feature
@app.route('/features/<int:n>', methods=['GET'])
def get_graph(n):
    G, pos = visualize_feature(n, activations, locations, tokens, tokenizer)
    nodes = [{"id": node, "x": pos[node][0], "y": pos[node][1]} for node in G.nodes()]
    edges = [{"source": u, "target": v} for u, v in G.edges()]
    return render_template("index.html", nodes=nodes, edges=edges)

if __name__ == "__main__":
    print(f'[MAIN] loading tokens and tokenizer')
    tokens, tokenizer = load_tokens("google/gemma-2-9B", "kh4dien/fineweb-100m-sample")
    print(f'[MAIN] loading activations and locations')
    activations, locations = load_activations('features/11_0_3275.safetensors')
    print(f'[MAIN] Done')
    app.run(debug=True)

    
