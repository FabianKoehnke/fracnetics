import React, { memo, useCallback } from 'react';
import { Handle, Position, NodeToolbar, useReactFlow, useConnection } from '@xyflow/react';
import "./index.css"

function gnpNode({ id, data }) {
  const connection = useConnection();
  const isTarget = connection.inProgress && connection.fromNode.id !== id;

  const { setNodes } = useReactFlow();

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

  if (data.label.slice(0,2) != "SN") {
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
        <NodeToolbar isVisible={data.forceToolbarVisible || undefined} position={data.toolbarPosition}>       
          <button onClick={() => changeNodeType('j')}>Judgment Node</button>
          <button onClick={() => changeNodeType('p')}>Processing Node</button>
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