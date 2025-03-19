import React, { memo, useCallback, useState, useEffect } from 'react';
import { Handle, Position, NodeToolbar, useReactFlow, useEdges } from '@xyflow/react';
import "./index.css"

function gnpNode({ id, data }) {
  const { setNodes, getEdges } = useReactFlow();
  const edges = getEdges();
  const outgoingEdgesCount = edges.filter(edge => edge.source === id).length;
  const [inputValue, setInputValue] = useState(data.storedValues ? data.storedValues.join(',') : '');

  useEffect(() => {
    if (data.storedValues) {
      setInputValue(data.storedValues.join(','));
    } else {
      setInputValue('');
    }
  }, [data.storedValues]);

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
          if (values.length == outgoingEdgesCount) {
            values = values;
          } else {
            values = ""
          }
          return {
            ...node,
            data: {
              ...node.data,
              storedValues: values,
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
            size="10"
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            placeholder="Enter value"
          />
        </NodeToolbar>      
      </div>
    );

  } else if (data.label.slice(0,2) == "JN") {
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
            size="27"
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            placeholder="Enter comma-separated values"
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