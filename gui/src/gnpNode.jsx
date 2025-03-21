import React, { memo, useCallback, useState, useEffect } from 'react';
import { Handle, Position, NodeToolbar, useReactFlow, useEdges } from '@xyflow/react';
import "./index.css"

function gnpNode({ id, data }) {
  const { setNodes, getEdges } = useReactFlow();
  const edges = getEdges();
  const outgoingEdgesCount = edges.filter(edge => edge.source === id).length;
  const [inputValue, setInputValue] = useState(data.nodeValues ? data.nodeValues.join(',') : '');

  useEffect(() => {
    if (data.nodeValues) {
      setInputValue(data.nodeValues.join(','));
    } else {
      setInputValue('');
    }
  }, [data.nodeValues]);

  const changeNodeType = useCallback(
    (gnpType) => {
      setNodes((nodes) =>
        nodes.map((node) => {
          if (node.id === id) {
            let newLabel = data.label;
            let newStyle = data.style;

            if (gnpType === 'j') {
              newLabel = `JN:${id}`;
              newStyle = 'judgmentNode'; 
            } else if (gnpType === 'p') {
              newLabel = `PN:${id}`;
              newStyle = 'processingNode'; 
            }

            return {
              ...node,
              data: {
                ...node.data,
                label: newLabel,
                style: newStyle,
              },
            };
          }
          return node;
        })
      );
    },
    [id, setNodes, data]
  );

  const handleInputChange = (event) => {
    const newValue = event.target.value;
    setInputValue(newValue);
  };

  useEffect(() => {
    setNodes((nodes) =>
      nodes.map((node) => {
        if (node.id === id) {
          let values = inputValue.split(',').map(v => v.trim()).filter(v => v !== '');
          
          if (data.label.slice(0,2) == "PN") {
            if (values.length <= 1) {
              values = values;
            } else {
              values = ""
            }
          } else if (data.label.slice(0,2) == "JN") {
            if (values.length == outgoingEdgesCount+1) {
              values = values;
            } else {
              values = ""
            }
          }

          return {
            ...node,
            data: {
              ...node.data,
              nodeValues: values,
            },
          };
        }
        return node;
      })
    );
  }, [inputValue, id, setNodes, outgoingEdgesCount]);

  if (data.label.slice(0,1) == "N") {
    return (
      <div className={`gnp-node-outer ${data.style}`} style={{backgroundColor: data?.color}} >
        <div className="gnp-node-inner" style={{backgroundColor: data?.color}} >
          {data?.label}
        </div>      
        <Handle
          type='source'
          position={Position.Bottom}
          onConnect={(params) => console.log('handle onConnect', params)}          
        />
        <Handle
          type='target'
          position={Position.Top}
          onConnect={(params) => console.log('handle onConnect', params)}
        />
        <NodeToolbar className="toolbar" isVisible={data.forceToolbarVisible || undefined} position={data.toolbarPosition}>       
          <button onClick={() => changeNodeType('j')}>Judgment Node</button>
          <button onClick={() => changeNodeType('p')}>Processing Node</button>
        </NodeToolbar>      
      </div>
    );

  } else if (data.label.slice(0,2) == "PN") {
    return (
      <div className={`gnp-node-outer ${data.style}`} style={{backgroundColor: data?.color}} >
        <div className="gnp-node-inner" style={{backgroundColor: data?.color}} >
          {data?.label}
        </div>      
        <Handle
          type='source'
          position={Position.Bottom}
          onConnect={(params) => console.log('handle onConnect', params)}          
        />
        <Handle
          type='target'
          position={Position.Top}
          onConnect={(params) => console.log('handle onConnect', params)}
        />
        <NodeToolbar className="toolbar" isVisible={data.forceToolbarVisible || undefined} position={data.toolbarPosition}>       
          <input
            size="11"
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            placeholder="Enter y-value"
          />
        </NodeToolbar>      
      </div>
    );

  } else if (data.label.slice(0,2) == "JN") {
    return (
      <div 
        className={`gnp-node-outer ${data.style}`} 
        style={{backgroundColor: data?.color}} 
        >
        <div 
          className={`gnp-node-inner ${data.style}`} 
          style={{backgroundColor: data?.color}} 
          >
          {data?.label}
        </div>      
        <Handle
          type='source'
          position={Position.Bottom}
          onConnect={(params) => console.log('handle onConnect', params)}          
        />
        <Handle
          type='target'
          position={Position.Top}
          onConnect={(params) => console.log('handle onConnect', params)}
        />
        <NodeToolbar className="toolbar" isVisible={data.forceToolbarVisible || undefined} position={data.toolbarPosition}>       
          <input
            size="31"
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            placeholder="Enter comma-separated boundaries"
          />
        </NodeToolbar>      
      </div>
    );

  } else if (data.label.slice(0,2) == "SN") {
      return (
        <div className={`gnp-node-outer ${data.style}`} style={{backgroundColor: data?.color}} >
          <div className="gnp-node-inner" style={{backgroundColor: data?.color}} >
            {data?.label}
          </div>      
          <Handle
            type='source'
            position={Position.Bottom}
            onConnect={(params) => console.log('handle onConnect', params)}
          />
        </div>
    );
  }
  
}

export default memo(gnpNode);