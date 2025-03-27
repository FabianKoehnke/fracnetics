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

    function findNodes(nodeType = ""){
        let nNodes = 0;
        for(let i=0; i<nodes.length; i++){
            if(nodes[i].data.label.slice(0,2) === nodeType){
                nNodes++;
            }
        }
        return nNodes;
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

    function getRandomFloat(min, max) {
        return Math.random() * (max - min) + min; // Hint Math.random: 0 (included) and 1 (not included)
      }

    function binarySearch(arr, target) {
        let left = 0;
        let right = arr.length - 1;

        if(target < arr[0]){
            return 0;
        } else if (target > arr[arr.length-1]){
            return arr.length-2;
        }
        
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
        
            if (arr[mid] <= target && arr[mid+1] > target) {
            return mid; 
            }
        
            if (arr[mid] < target) {
            left = mid + 1; 
            } else {
            right = mid - 1; 
            }
        }          
        return -1;
        }

    function runTransitionPath(n){    
        
        const judgmentNodeFunctions = findJudgmentnodeFunctions();   
        const [minValues, maxValues] = findMinAndMaxOfBoundaries(judgmentNodeFunctions);  
        
        // initialize data set 
        const dataTarget = new Array(n) 
        const dataFeatures = new Array(n)
        for(let i=0; i<n; i++){
            dataFeatures[i] = new Array(judgmentNodeFunctions.length).fill(0)
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
            let nProcessings = findNodes("PN");
            let nUndefinedNodes = findNodes("N ")
            if(nProcessings>0 && nUndefinedNodes <0){
                let i = 0;
                let nodeFunction = undefined;
                let nodeValues = undefined;
                let idx = undefined;
                let randomFloat = undefined;
                let idxBoundaries = [];
                while(i < n) {
                    const usedJudgmentNodeFunctions = [];
                    if(nodes[currentNodeID].data.label.slice(0,2) == "PN"){
                        dataTarget[i] = parseFloat(nodes[currentNodeID].data.nodeValues[0]); 
                        // set values for unused judgment nodes
                        if(judgmentNodeFunctions.length>0){
                            for(let j=0; j<judgmentNodeFunctions.length;j++){
                                if(!usedJudgmentNodeFunctions.includes(judgmentNodeFunctions[j])){
                                    nodeFunction = judgmentNodeFunctions[j];
                                    usedJudgmentNodeFunctions.push(nodeFunction);
                                    idx = judgmentNodeFunctions.indexOf(nodeFunction);
                                    randomFloat = getRandomFloat(minValues[idx],maxValues[idx]); // TODO change min and max to given boundary
                                    dataFeatures[i][idx] = randomFloat;
                                }
                            }
                            usedJudgmentNodeFunctions.length = 0; 
                        } 
                        
                        currentNodeID = nodes[currentNodeID].data.edges[0].target; // set new node
                        i++;
                    } else if(nodes[currentNodeID].data.label.slice(0,2) == "JN") {                        
                        nodeFunction = nodes[currentNodeID].data.nodeFunction;
                        nodeValues = nodes[currentNodeID].data.nodeValues;
                        usedJudgmentNodeFunctions.push(nodeFunction);
                        idx = judgmentNodeFunctions.indexOf(nodeFunction);
                        randomFloat = getRandomFloat(nodeValues[0],nodeValues[nodeValues.length-1]); // TODO change min and max to given boundary
                        idxBoundaries = binarySearch(
                            nodeValues.map(element => parseFloat(element)), 
                            randomFloat
                            );
                        currentNodeID = nodes[currentNodeID].data.edges[idxBoundaries].target; // set new node
                        dataFeatures[i][idx] = randomFloat;
        
                    } else if(nodes[currentNodeID].data.label.slice(0,2) == "SN") {
                        currentNodeID = nodes[currentNodeID].data.edges[0].target; // set new node
                    }
                }
            }

        } catch(err) {
            console.log(`error in transition path: ${err.message}`);
        }

        return [dataTarget, dataFeatures];
    }

    let n = 100; // number of datapoints (reached processing nodes)       
    let data = runTransitionPath(n);
    console.log('Aktuelle Knoten:', nodes);
    console.log('Aktuelle Kanten:', edges);
    return data;
  }