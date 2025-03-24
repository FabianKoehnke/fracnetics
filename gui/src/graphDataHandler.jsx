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
                if(judgmentNodeFunctions.includes(nodes[i].data.nodeFunctions)){
                    continue;
                } else{                                        
                    judgmentNodeFunctions.push(nodes[i].data.nodeFunctions);
                }
            }            
        }
        return judgmentNodeFunctions;
    }

    function runTransitionPath(n){    
        
        const judgmentNodeFunctions = findJudgmentnodeFunctions();        
        
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
            for(let i = 0; i < n+1; i++) { // n+1 because of start node
                if(nodes[currentNodeID].data.label.slice(0,2) == "PN"){
                    currentNodeID = nodes[currentNodeID].data.edges[0].target; // set new node
                
                } else if(nodes[currentNodeID].data.label.slice(0,2) == "JN") {
                    
    
                } else if(nodes[currentNodeID].data.label.slice(0,2) == "SN") {
                    currentNodeID = nodes[currentNodeID].data.edges[0].target; // set new node
                }
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