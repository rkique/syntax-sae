document.addEventListener("DOMContentLoaded", () => {
    const graphDiv = document.getElementById("graphDiv");

    // Extract node positions
    const x = nodes.map(node => node.x);
    const y = nodes.map(node => node.y);
    const nodeIds = nodes.map(node => node.id);

    // Define nodes for Plotly
    const nodeTrace = {
        x: x,
        y: y,
        text: nodeIds,
        mode: 'markers+text',
        marker: {
            size: 10,
            color: 'blue',
        },
        type: 'scatter',
        textposition: 'top center',
        hoverinfo: 'text'
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

    // Plot graph
    Plotly.newPlot(graphDiv, [edgeTrace, nodeTrace], layout);
});