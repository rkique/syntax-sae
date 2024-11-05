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
        const childXOffset = 60; // Horizontal distance between siblings
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

        const maxTagValue = Math.max(...nodes.map(node => node.tag));

        const getColorGradient = (tag) => {
            const startColor = { r: 255, g: 255, b: 255 }; // Orange for tag 1
            const endColor = { r: 0, g: 0, b: 255 }; // Blue for tag 0
            const ratio = maxTagValue > 0 ? tag / maxTagValue : 0; // Avoid division by zero
            // Clamp the ratio between 0 and 1
            const clampedRatio = Math.max(0, Math.min(1, ratio));
            // Calculate the color based on the ratio
            const r = Math.round(startColor.r + clampedRatio * (endColor.r - startColor.r));
            const g = Math.round(startColor.g + clampedRatio * (endColor.g - startColor.g));
            const b = Math.round(startColor.b + clampedRatio * (endColor.b - startColor.b));
        
            return `rgb(${r}, ${g}, ${b})`;
        };
        const nodeColors = nodes.map(node => getColorGradient(node.tag)); // Color logic
        // Create a container for each graph and its corresponding context
        const graphContextContainer = document.createElement("div");
        graphContextContainer.classList.add("graphContextContainer"); // Add a class for styling

        const graphDiv = document.createElement("div");
        graphDiv.id = `graphDiv-${index}`; // Unique ID for each graph
        graphDiv.classList.add("graphDiv"); 
        graphContextContainer.appendChild(graphDiv);

        const contextDiv = document.createElement("div");
        contextDiv.id = `contextDiv-${index}`; // Unique ID for each context
        contextDiv.classList.add("contextDiv"); 
        activationDict = activation_dicts[index]
        const maxActivation = Math.max(...Object.values(activationDict));
        // Create the context with color highlighting
        contextDiv.innerHTML = contexts[index].map((word, index) => {
            // Get activation level or default to 0
            const activation = activationDict[index] || 0;
            // Calculate blue intensity from 0 (white) to 255 (blue) based on activation
            const blueIntensity = Math.round((activation / maxActivation) * 255);
            const color = `rgb(${255 - blueIntensity}, ${255 - blueIntensity}, 255)`;
            const textColor = blueIntensity > 127 ? "white" : "black";
            return `<span style="background-color: ${color}; color: ${textColor}; padding: 2px;">${word}</span>`;
        }).join(" ");

        graphContextContainer.appendChild(contextDiv);
        graphContainer.appendChild(graphContextContainer);

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