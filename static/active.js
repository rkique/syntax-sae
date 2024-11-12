document.addEventListener("DOMContentLoaded", () => {
    let data = [
        [
          [" MS", 3.5826505059221563],
          ["ps", 2.372723141848354],
          [" trust", 2.25866585905921]
        ],
        [
          [" deep", 2.5745188084776873],
          [" south", 2.5631448456936776],
          ["-", 2.4585198946148052]
        ],
        [
          [" Windows", 7.083938241432844],
          [" Southern", 5.804946698182618],
          [" southern", 5.803486396912678]
        ],
        [
          [" XP", 4.336268977179685],
          [" accent", 3.5407328491184433],
          [" hospitality", 3.20566393089597]
        ],
        [
          ["2", 3.3004558118606235],
          [" Mexico", 3.273008833448128],
          [" States", 3.2005845548959493]
        ]
      ];
      
      let margin = 50;
      let barWidth = 100;
      let maxBarHeight = 500;
      
      function setup() {
        let canvas = createCanvas(600, 600);
        canvas.parent('chart-container');
        noStroke();
        textAlign(CENTER, CENTER);
      }
      
      function draw() {
        background(255);
        
        // Loop through each "group" in the data array
        for (let i = 0; i < data.length; i++) {
          let yOffset = 0; // To stack the rectangles
      
          // Loop through each "layer" (data point) for the group
          for (let j = 0; j < data[i].length; j++) {
            let label = data[i][j][0];
            let value = data[i][j][1];
            
            // Set a color for each layer
            fill(map(j, 0, data[i].length, 100, 255), 100, 200);
      
            // Draw the rectangle (stacked bar)
            rect(margin + i * (barWidth + 20), height - margin - yOffset - value, barWidth, value);
      
            // Draw the label text in the middle of the rectangle
            fill(0);
            text(label, margin + i * (barWidth + 20) + barWidth / 2, height - margin - yOffset - value / 2);
      
            // Update yOffset to stack the next rectangle
            yOffset += value;
          }
        }
      }
});