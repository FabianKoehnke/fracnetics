import randomNormal from 'random-normal';

export function handleGraphData(nodes, edges) {
    /**
     * This method handle the data of the current graph.
     * 
     * If the connection of nodes and edges are valid,
     * then the transition path is executed and returned.
     * 
     * @param {Object} nodes 
     * @param {Object} edges 
     * 
     * @returns {Number[][]} Data of transition path
     */
    
    function findJudgmentnodeFunctions(){
        /**
         * This method finds all distinct judgment node functions.
         * 
         * @returns {Array} distinced judgment node functions
         */
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
        /**
         * This methods counts the number of nodes in the network
         * of given node type.
         * 
         * @param {String} nodeType
         * - "PN": processing node
         * - "JN": judgment node
         * - "N": undefined node
         * 
         * @returns {Number} counted nodes
         */
        let nNodes = 0;
        for(let i=0; i<nodes.length; i++){
            if(nodes[i].data.label.slice(0,nodeType.length) === nodeType){
                nNodes++;
            }
        }
        return nNodes;
    }

    function findMinAndMaxOfBoundaries(judgmentNodeFunctions){
        /**
         * This method finds the min. and max. values of all distinct
         * judgment node functions. 
         * This is necessary for drawing feature values of not visited
         * judgment node functions during the transition path between two 
         * processing nodes. Otherwise there would not a completly filled 
         * dataset. 
         * 
         * @param {Array} judgmentNodeFunctions - disticet judgment node functions
         * @returns {Array, Array} [minValues, maxValues]
         */
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
        /**
         * This methods finds the index of an array given
         * a value via binary search.
         * It is used for the judgment of a judgment node 
         * given there interval boundaries and a feature value.
         * 
         * @param {Array} arr - GNP: interval boundaries
         * @param {Number} target
         * 
         * @returns {Number} index of given array
         */
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
        /**
         * This method runs through the transition path.
         * 
         * @param {Number} n - number of visit processing nodes (data rows)
         * @returns {Array, Array} [dataTarget, dataFeatures]
         */
        
        const judgmentNodeFunctions = findJudgmentnodeFunctions();   
        const [minValues, maxValues] = findMinAndMaxOfBoundaries(judgmentNodeFunctions);  
        
        // initialize data set 
        const dataTarget = new Array(n);
        const dataFeatures = new Array(n);
        for(let i=0; i<n; i++){
            dataFeatures[i] = new Array(judgmentNodeFunctions.length).fill(0);
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
        let currentNodeID = nodes[0].id;
        try{
            let nProcessings = findNodes("PN");
            let nUndefinedNodes = findNodes("N")
            if(nProcessings > 0 && nUndefinedNodes === 0){
                let i = 0;
                let nodeFunction = undefined;
                let nodeValues = undefined;
                let idx = undefined;
                let randomFloat = undefined;
                let idxBoundaries = [];
                let distribution = undefined;
                while(i < n) {
                    const usedJudgmentNodeFunctions = [];
                    if(nodes[currentNodeID].data.label.slice(0,2) == "PN"){ // node is a processing node
                        distribution = nodes[currentNodeID].data.distribution;
                        if(distribution == "Uniform"){
                            dataTarget[i] = parseFloat(nodes[currentNodeID].data.nodeValues[0]); 
                        } else if(distribution == "Normal"){
                            let mean = parseFloat(nodes[currentNodeID].data.nodeValues[0]);
                            dataTarget[i] = parseFloat(randomNormal({mean: mean, dev: 0.01}));
                        }
                        
                        // set values for unused judgment nodes
                        if(judgmentNodeFunctions.length>0){
                            for(let j=0; j<judgmentNodeFunctions.length;j++){
                                if(!usedJudgmentNodeFunctions.includes(judgmentNodeFunctions[j])){
                                    nodeFunction = judgmentNodeFunctions[j];
                                    usedJudgmentNodeFunctions.push(nodeFunction);
                                    idx = judgmentNodeFunctions.indexOf(nodeFunction);
                                    randomFloat = getRandomFloat(minValues[idx],maxValues[idx]);
                                    if(distribution == "Uniform"){
                                        randomFloat = getRandomFloat(parseFloat(minValues[idx]),parseFloat(maxValues[idx]));
                                    } else if(distribution == "Normal"){
                                        let mean = parseFloat(minValues[idx]) + ((parseFloat(maxValues[idx]) - parseFloat(minValues[idx])) / 2);
                                        randomFloat = randomNormal({mean: mean, dev: 1});
                                    }
                                    dataFeatures[i][idx] = randomFloat;
                                }
                            }
                            usedJudgmentNodeFunctions.length = 0; 
                        } 
                        
                        currentNodeID = nodes[currentNodeID].data.edges[0].target; // set new node
                        i++;
                    } else if(nodes[currentNodeID].data.label.slice(0,2) == "JN") { // node is a judgment node                  
                        nodeFunction = nodes[currentNodeID].data.nodeFunction;
                        nodeValues = nodes[currentNodeID].data.nodeValues;
                        distribution = nodes[currentNodeID].data.distribution;
                        usedJudgmentNodeFunctions.push(nodeFunction);
                        idx = judgmentNodeFunctions.indexOf(nodeFunction);
                        if(distribution == "Uniform"){
                            randomFloat = getRandomFloat(parseFloat(nodeValues[0]),parseFloat(nodeValues[nodeValues.length-1]));
                        } else if(distribution == "Normal"){
                            let mean = parseFloat(nodeValues[0]) + ((parseFloat(nodeValues[nodeValues.length-1]) - parseFloat(nodeValues[0])) / 2);
                            randomFloat = randomNormal({mean: mean, dev: 1});
                        }
                        
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
            dataTarget.fill(undefined)
            console.log(`error in transition path: ${err.message}`);
        }

        return [dataTarget, dataFeatures];
    }

    let n = 1000; // number of datapoints (reached processing nodes)       
    let data = runTransitionPath(n);
    if(data[0][0] === undefined || isNaN(data[0][0])){
        data = null;
    }
    console.log('Aktuelle Knoten:', nodes);
    console.log('Aktuelle Kanten:', edges);
    return data;
  }