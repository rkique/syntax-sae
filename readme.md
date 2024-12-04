# Syntax-based Contextual Visualizations for SAE & LLM Interpretability

## Project Overview
This project aims to improve interpretability measures by developing a new visualization method for Sparse Autoencoder (SAE) feature contexts. Specifically, we proposed using syntactic dependencies to illuminate similarities between contexts.

## Project Features
We were able to develop three novel views for activation contexts, two of which utilize syntactic dependency structures and one which uses branching trees. We use the SpaCy dependency parser and sentence tagger on the backend. These new views are meant to supplement activation context lists, e.g. those developed by Anthropic:

[](images/anthropic.png)

The joint view shows individual contexts side by side. You can enable part of speech tagging or view inactive tokens through the top panel:

[](images/joint.png)

The merged view aggregates commonly occurring contexts and displays them in a branching format. These trees are instantiated as list structures and subtree matches are located where possible.

[](images/merged.png)

## Acknowledgements

This work was done for David Laidlaw's CSCI2370: Interdisciplinary Scientific Visualization class.