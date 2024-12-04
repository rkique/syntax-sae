
let activeLevel = 1;
let HEIGHT = 800
let yMin = -5000 / 4
let yMax = 500 / 4

const initialLayout = (activeLevel, backgroundShapes) => {
    return {
    shapes: backgroundShapes,
    font: {
        size: 8 // Font size for the title
    },
    xaxis: {
        showgrid: true,
        zeroline: false,
        showticklabels: false,
        title: '', // You can add titles if needed
        automargin: true, // Automatically adjust margins
        range: [-1000, 3000]
    },
    yaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: true,
        title: '', // You can add titles if needed
        automargin: true, // Automatically adjust margins
        range: [yMin,yMax],
    },
    showlegend: false,
    margin: {
        l: 10,   // Left margin
        r: 10,   // Right margin
        b: 10,   // Bottom margin
        t: 70    // Top margin
    },
    width: 1200, // Make the plot use full width of the container
    height: HEIGHT, // Make the plot use full height of the container
    dragmode: 'pan',
    sliders: [
            {
            yanchor: 'top',    // Anchor the slider to the top
            y: 1.15,           // Position the slider above the graph (relative to layout height)
            pad: { t: 0 },     // Adjust padding for positioning
            active: activeLevel,
            currentvalue: {
                visible: true,
                prefix: 'Zoom: ',
                //xanchor: 'center',
                font: {
                    size: 12
                }
            },
            steps: []
        }
    ]
    };
}


document.addEventListener("DOMContentLoaded", () => {
    const graphContainer = document.getElementById("graphContainer"); // Parent container for all graphs
    const IdCheckbox = document.getElementById("toggleIdFormat");
    //const OmitCheckbox = document.getElementById("toggleOmitPOS");
    const ZeroCheckbox = document.getElementById("toggleZeroFormat");
    
    var idCounter = 0;
    const assignUniqueIds = (node) => {
        idCounter++;
        node.id = `n${idCounter}`; 
        if (Array.isArray(node.children)) {
            node.children.forEach(child => assignUniqueIds(child));
        }
    };

    const text_for_node = (node) => {
        //console.log(node)
        if (IdCheckbox.checked) {
            //console.log('toggle is checked')
            return `${[node.pos]}`;
        } else {
            if (node.hasOwnProperty('occurrences') && node.occurrences > 1) {
            //console.log('toggle is not checked')
            return `${node.text} (${node.occurrences})`;
            }
            else {
                return `${node.text}`
            }
        }
    };

const extractNodesAndEdges = (node, nodes, edges, x = 0, y = 0, depth = 0) => {
    // Start: put root at (0, 0)
    if (depth == 1){
    console.log(x,y, depth)
    }
    nodes.push({ 
        id: node.id, 
        zero: ZeroCheckbox.checked ? (node.tag === 0) : false, 
        text: text_for_node(node), 
        x: x, 
        y: y, 
        tag: node.tag
    });

    // Get children of the current node
    const children = Array.isArray(node.children) ? node.children : [];
    const childXStart = x + 200; 
    const baseChildYOffset = 300; 
    const childYOffset = baseChildYOffset / (1 + depth * 3); 
    const totalChildHeight = (children.length - 1) * childYOffset; 
    let childYStart = (y - totalChildHeight / 2); // Center children around the current node
    if (depth == 0){
        childYStart *= 2
    }
    children.forEach(child => {
        const edge = { source: node.id, target: child.id };
        edge.zero = ZeroCheckbox.checked ? (node.tag === 0 || child.tag === 0) : false;
        edges.push(edge);
        // Recursive call for each child with updated depth
        extractNodesAndEdges(child, nodes, edges, childXStart, childYStart, depth + 1);
        childYStart += childYOffset;
    });
};

    const displayNodes = (activeLevel) => {
        const existingContainers = document.querySelectorAll(".graphContextContainer");
        existingContainers.forEach(container => container.remove());
    
        graphs.forEach((graph, index) => {
            console.log('resetting nodes')
            var nodes = [];
            var edges = [];

            idCounter = 0
            assignUniqueIds(graph);
            extractNodesAndEdges(graph, nodes, edges);

            const maxTagValue = Math.max(...nodes.map(node => node.tag));

            const getColorGradient = (node) => {
                const startColor = { r: 255, g: 255, b: 255 };1
                const endColor = { r: 0, g: 0, b: 255 };
                const ratio = maxTagValue > 0 ? node.tag / maxTagValue : 0; 
                
                const clampedRatio = Math.max(0, Math.min(1, ratio));
                const r = Math.round(startColor.r + clampedRatio * (endColor.r - startColor.r));
                const g = Math.round(startColor.g + clampedRatio * (endColor.g - startColor.g));
                const b = Math.round(startColor.b + clampedRatio * (endColor.b - startColor.b));
                
                return `rgb(${r}, ${g}, ${b})`;
            };
            // Create a container for each graph and its corresponding context
            const graphContextContainer = document.createElement("div");
            graphContextContainer.classList.add("graphContextContainer"); // Add a class for styling

            const graphDiv = document.createElement("div");
            graphDiv.id = `graphDiv-${index}`;
            graphDiv.classList.add("graphDiv"); 
            graphContextContainer.appendChild(graphDiv);
           
            // graphContextContainer.appendChild(contextDiv);
            graphContainer.appendChild(graphContextContainer);

            const filteredNodes = nodes.filter(node => !node.zero);

            const x = filteredNodes.map(node => node.x);
            const y = filteredNodes.map(node => node.y);
            const nodeIds = filteredNodes.map(node => node.text);
            const nodeColors = filteredNodes.map(node => getColorGradient(node));
            
            // Define nodes for Plotly
            const nodeTrace = {
                x: x,
                y: y,
                text: nodeIds.map((id, index) => `<span style="background-color: ${nodeColors[index]};">${id}</span>`), // Dynamic color per id
                mode: 'text',
                // marker: {
                //     size: 10,
                //     color: nodeColors,
                // },
                type: 'scatter',
                textposition: 'center',  
                hoverinfo: 'none' 
            };

            // Define a function to create rectangles as text backgrounds
            const createBackgroundShapes = (x, y, nodeIds, nodeColors, textFontSize) => {
                const rectBuffer = textFontSize * 0.1; // Adjust buffer size (proportional to font size)
                const charWidth = textFontSize * 0.6; // Approximate width per character
                const textHeight = textFontSize * 1.2; // Approximate height of text

                return nodeIds.map((id, index) => ({
                    type: 'rect',
                    xref: 'x',
                    yref: 'y',
                    x0: x[index] - (charWidth * id.length) / 2 - rectBuffer, // Left boundary
                    x1: x[index] + (charWidth * id.length) / 2 + rectBuffer, // Right boundary
                    y0: y[index] - textHeight / 2 - rectBuffer,              // Bottom boundary
                    y1: y[index] + textHeight / 2 + rectBuffer,              // Top boundary
                    fillcolor: nodeColors[index], // Match node color
                    opacity: 0.5,                // Background transparency
                    line: {
                        width: 0                // No border
                    }
                }));
            };

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

            edges.forEach(edge => {
                if (!edge.zero) {
                    const sourceNode = nodes.find(node => node.id === edge.source);
                    const targetNode = nodes.find(node => node.id === edge.target);
                    edgeTrace.x.push(sourceNode.x, targetNode.x, null);
                    edgeTrace.y.push(sourceNode.y, targetNode.y, null);
                }
            });
            const config = {
                displayModeBar: false // Disables the modebar
            };

            const textFontSize = 12; // Same font size as in node trace
            const backgroundShapes = createBackgroundShapes(x, y, nodeIds, textFontSize, nodeColors);

            // Plotly layout configuration
            const layout = initialLayout(activeLevel, backgroundShapes)
            // const zoomLevels = [0.5, 1, 2, 4, 8, 16]; 
            //this slider should update y axis around the current level.
            // zoomLevels.forEach((scale) => {
            //     console.log(`${layout.sliders[0].currentvalue}`)
            //     layout.sliders[0].steps.push(
            //         {
            //         method: 'relayout',
            //         args: ['yaxis.range', [-5000, 1000]],
            //         label: `x${scale}`
            //     }
            //     );
            // });
            // Plot the graph with the slider
            Plotly.newPlot(graphDiv, [nodeTrace, edgeTrace], layout, config);
                    });
                }
    displayNodes(4)

    // OmitCheckbox.addEventListener('change', () => {
    //     omitPOS = OmitCheckbox.checked ? [97, 99, 101, 102, 103] : []
    //     applyHighlighting(statistics, omitPOS)
    // });
    
    graphDiv = document.getElementById(`graphDiv-0`)
    graphDiv.on('plotly_sliderchange', (eventData) => {
        // Access the updated layout from the event data
        console.log(eventData.step._index)
        activeLevel = eventData.step._index
    });

    IdCheckbox.addEventListener('change', () => {
        console.log(`${activeLevel}`)
        displayNodes(activeLevel);
    });

    ZeroCheckbox.addEventListener('change', () => {
        console.log(`${activeLevel}`)
        displayNodes(activeLevel);
    });
});

