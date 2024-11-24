
let activeLevel = 1;
const initialLayout = (activeLevel) => {
    return {
    font: {
        size: 8 // Font size for the title
    },
    xaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        title: '', // You can add titles if needed
        automargin: true, // Automatically adjust margins
        range: [-10000, 10000]
    },
    yaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        title: '', // You can add titles if needed
        fixedrange: true,
        automargin: true // Automatically adjust margins
    },
    showlegend: false,
    margin: {
        l: 10,   // Left margin
        r: 10,   // Right margin
        b: 10,   // Bottom margin
        t: 70    // Top margin
    },
    width: 1200, // Make the plot use full width of the container
    height: 800, // Make the plot use full height of the container
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
                xanchor: 'center',
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
    // Function to traverse the graph and extract nodes and edges
    const extractNodesAndEdges = (node, nodes, edges, x = 0, y = 0) => {
        nodes.push({ 
            id: node.id, 
            zero: ZeroCheckbox.checked ? (node.tag === 0) : false, 
            text: text_for_node(node), 
            x: x,
            y: y, 
            tag: node.tag
        });
        const children = Array.isArray(node.children) ? node.children : [];
        const childXStart = x + 200;  // Adjusted child position for x
        const childYOffset = 1000  // Adjusted child offset for y
        let childY = y + (children.length - 1) * (childYOffset / 3);  // Adjusted starting position for y
        children.forEach(child => {
            const edge = { source: node.id, target: child.id };
            if (ZeroCheckbox.checked) {
                edge.zero = (node.tag === 0 || child.tag === 0);
            } else {
                edge.zero = false;
            }
            edges.push(edge);
            // Recursive call with switched coordinates
            extractNodesAndEdges(child, nodes, edges, childXStart, childY);
            childY -= childYOffset;
        });
    };
    const displayNodes = (activeLevel) => {
        const existingContainers = document.querySelectorAll(".graphContextContainer");
        existingContainers.forEach(container => container.remove());
    
        // Iterate over each graph in the graphs array
        graphs.forEach((graph, index) => {
            console.log('resetting nodes')
            var nodes = [];
            var edges = [];

            // Start extraction from the root node
            idCounter = 0
            assignUniqueIds(graph);
            extractNodesAndEdges(graph, nodes, edges);

            const maxTagValue = Math.max(...nodes.map(node => node.tag));

            const getColorGradient = (node) => {
                // const keywords = ['recalls', 'recall', 'Recall', 'recalls', 'Recalled', 'Recalls', 'recalling', 'recalled']
                // if (keywords.includes(node.text)){
                //     return `rgb(255, 255, 0)`;
                // }
                const startColor = { r: 255, g: 255, b: 255 }; // Orange for tag 1
                const endColor = { r: 0, g: 0, b: 255 }; // Blue for tag 0
                const ratio = maxTagValue > 0 ? node.tag / maxTagValue : 0; // Avoid division by zero
                
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

            // const contextDiv = document.createElement("div");
            // contextDiv.id = `contextDiv-${index}`;
            // contextDiv.classList.add("contextDiv"); 
            // activationDict = activation_dicts[index]
            // console.log(activationDict)
            // const maxActivation = 100;
            
            // Create the context with color highlighting
            // contextDiv.innerHTML = contexts[index].map((word, index) => {
            //     const activation = activationDict[index] || 0;
            //     const blueIntensity = Math.round((activation / maxActivation) * 255);
            //     const color = `rgb(${255 - blueIntensity}, ${255 - blueIntensity}, 255)`;
            //     const textColor = blueIntensity > 127 ? "white" : "black";
            //     return `<span style="background-color: ${color}; color: ${textColor}; padding: 2px;">${word}</span>`;
            // }).join(" ");

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
                text: nodeIds,  // Labels for each node
                mode: 'markers+text',
                marker: {
                    size: 10,
                    color: nodeColors,
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
            
            // Plotly layout configuration
            const layout = initialLayout(activeLevel)
            // Define zoom levels and slider     steps
            const zoomLevels = [20, 40,80,160,320,640]; // Different zoom levels (scale factors)
            zoomLevels.forEach((scale) => {
                layout.sliders[0].steps.push({
                    method: 'relayout',
                    args: ['xaxis.range', [- 100000 / scale, 100000 / scale]], // Adjust range based on scale
                    label: `x${scale}`
                });
            });
            // Plot the graph with the slider
            Plotly.newPlot(graphDiv, [edgeTrace, nodeTrace], layout, config);
                    });
                }
    displayNodes(3)

    var omitPOS = [97, 99, 101, 102, 103]

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

