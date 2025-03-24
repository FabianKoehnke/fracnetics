/**
 * This method handle the data of the current graph
 * 
 * @param {*} nodes 
 * @param {*} edges 
 */
export function handleGraphData(nodes, edges) {
    
    function findJudgmentnodeFunctions(){
        const judgmentNodeFunctions = [];
        for(let i=0; i<nodes.length; i++){
            if (nodes[i].data.label.slice(0,2) == "JN") {
                if(judgmentNodeFunctions.includes(nodes[i].data.nodeFunction)){
                    continue;
                } else{                                        
                    judgmentNodeFunctions.push(nodes[i].data.nodeFunction);
                }
            }            
        }
        return judgmentNodeFunctions;
    }

    function findMinAndMaxOfBoundaries(judgmentNodeFunctions){
        const minValues = Array(judgmentNodeFunctions.length);
        const maxValues = Array(judgmentNodeFunctions.length);
        for(let i=0; i<nodes.length; i++){
            if(nodes[i].data.label.slice(0,2) == "JN"){
                let nodeFunction = nodes[i].data.nodeFunction;
                let idx = judgmentNodeFunctions.indexOf(nodeFunction);
                
                if (minValues[idx] === undefined || 
                        nodes[i].data.nodeValues[0] < minValues[idx]){                    
                    minValues[idx] = parseFloat(nodes[i].data.nodeValues[0]);
                }
                
                if (maxValues[idx] === undefined || 
                        nodes[i].data.nodeValues[nodes[i].data.nodeValues.length-1] > maxValues[idx]){                    
                    maxValues[idx] = parseFloat(nodes[i].data.nodeValues[nodes[i].data.nodeValues.length-1]);
                }
            }
        }
        return [minValues, maxValues]
    }

    function runTransitionPath(n){    
        
        const judgmentNodeFunctions = findJudgmentnodeFunctions();   
        const [minValues, maxValues] = findMinAndMaxOfBoundaries(judgmentNodeFunctions);  
        
        // initialize data set 
        let dataframe = new Array(n)
        for(let i=0; i<n; i++){
            dataframe[i] = new Array(judgmentNodeFunctions.length).fill(0)
        }
        
        // initialize edges in node object 
        for(let i = 0; i < nodes.length; i++){     
            nodes[i].data.edges = [];       
            for(let k = 0; k < edges.length; k++){
                if (edges[k].source == nodes[i].id){
                    nodes[i].data.edges.push(edges[k]);
                }
            }   
        }   

        // run transition path
        let currentNodeID = nodes[0].id
        try{
            let i = 0;
            while(i < n) {
                if(nodes[currentNodeID].data.label.slice(0,2) == "PN"){
                    currentNodeID = nodes[currentNodeID].data.edges[0].target; // set new node
                    
                } else if(nodes[currentNodeID].data.label.slice(0,2) == "JN") {
                        // draw a random value given pattern
    
                } else if(nodes[currentNodeID].data.label.slice(0,2) == "SN") {
                    currentNodeID = nodes[currentNodeID].data.edges[0].target; // set new node
                }
                i++;
            }

        } catch {
            console.log('Error in transition path');
        }
    }

    let n = 100; // number of datapoints (reached processing nodes)        
    runTransitionPath(n);
    console.log('Aktuelle Knoten:', nodes);
    console.log('Aktuelle Kanten:', edges);
    
  }