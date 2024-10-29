document.addEventListener("DOMContentLoaded", () => {
    const graphContainer = document.getElementById("graphContainer"); // Parent container for all graphs
    const id_for_node = (node) => {
        return `${node.text} [${node.pos}]`;
    }
    // Function to traverse the graph and extract nodes and edges
    const extractNodesAndEdges = (node, nodes, edges, x = 0, y = 0) => {
        // Add the current node
        nodes.push({ id: id_for_node(node), x: x, y: y, tag: node.tag});
        const children = Array.isArray(node.children) ? node.children : [];
        // Calculate child positions
        const childYStart = y - 50; // Vertical distance between parent and children
        const childXOffset = 50; // Horizontal distance between siblings
        let childX = x - (children.length - 1) * (childXOffset / 2); // Center children
        children.forEach(child => {
            // Add edge from current node to child
            edges.push({ source: id_for_node(node), target: id_for_node(child)});
            // Recursive call to process the child
            extractNodesAndEdges(child, nodes, edges, childX, childYStart);
            // Update the x position for the next child
            childX += childXOffset;
        });
    };

    // Iterate over each graph in the graphs array
    graphs.forEach((graph, index) => {
        const nodes = [];
        const edges = [];

        // Start extraction from the root node
        extractNodesAndEdges(graph, nodes, edges);

        // Extract node positions
        const x = nodes.map(node => node.x);
        const y = nodes.map(node => node.y);
        const nodeIds = nodes.map(node => node.id);
        const nodeColors = nodes.map(node => (node.tag === 1 ? 'orange' : 'blue')); // Color logic
        // Create a new div for the current graph
        const graphDiv = document.createElement("div");
        graphDiv.id = `graphDiv-${index}`; // Unique ID for each graph
        graphDiv.classList.add("graphDiv"); // Add the graphDiv class
        graphContainer.appendChild(graphDiv); // Append to the container

        // Define nodes for Plotly
        const nodeTrace = {
            x: x,
            y: y,
            text: nodeIds,  // Labels for each node
            mode: 'markers+text',
            marker: {
                size: 10,
                color: nodeColors,
            },
            type: 'scatter',
            textposition: 'top center',  // Position of the text relative to the markers
            hoverinfo: 'text'  // Show text on hover
        };

        // Define edges for Plotly
        const edgeTrace = {
            x: [],
            y: [],
            mode: 'lines',
            line: {
                width: 1,
                color: '#888'
            },
            type: 'scatter',
            hoverinfo: 'none'
        };

        // Add each edge as a line segment between source and target nodes
        edges.forEach(edge => {
            const sourceNode = nodes.find(node => node.id === edge.source);
            const targetNode = nodes.find(node => node.id === edge.target);
            edgeTrace.x.push(sourceNode.x, targetNode.x, null);
            edgeTrace.y.push(sourceNode.y, targetNode.y, null);
        });

        // Plotly layout configuration
        const layout = {
            xaxis: { showgrid: false, zeroline: false, showticklabels: false },
            yaxis: { showgrid: false, zeroline: false, showticklabels: false },
            showlegend: false,
        };

        // Plot the graph
        Plotly.newPlot(graphDiv, [edgeTrace, nodeTrace], layout);
    });
});