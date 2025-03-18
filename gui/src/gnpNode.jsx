import React, { memo, useCallback } from 'react';
import { Handle, Position, NodeToolbar, useReactFlow } from '@xyflow/react';
import "./index.css"

function gnpNode({ id, data }) {
  const { setNodes } = useReactFlow();

  const changeNodeType = useCallback(
    (gnpType) => {
      setNodes((nodes) =>
        nodes.map((node) => {
          if (node.id === id) {
            let newLabel = data.label;
            let newColor = data.color;

            if (gnpType === 's') {
              newLabel = `SN:${id}`;
              newColor = 'green';
            } else if (gnpType === 'j') {
              newLabel = `JN:${id}`;
              newColor = 'blue'; 
            } else if (gnpType === 'p') {
              newLabel = `PN:${id}`;
              newColor = 'orange'; 
            }

            return {
              ...node,
              data: {
                ...node.data,
                label: newLabel,
                color: newColor,
              },
            };
          }
          return node;
        })
      );
    },
    [id, setNodes, data]
  );

  return (
    <div className="gnp-node-outer" style={{backgroundColor: data?.color}} >
      <div className="gnp-node-inner" style={{backgroundColor: data?.color}} >
        {data?.label}
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        onConnect={(params) => console.log('handle onConnect', params)}
      />
      <Handle
        type="target"
        position={Position.Top}
        onConnect={(params) => console.log('handle onConnect', params)}
      />
      <NodeToolbar isVisible={data.forceToolbarVisible || undefined} position={data.toolbarPosition}>
        <button onClick={() => changeNodeType('s')}>Start Node</button>
        <button onClick={() => changeNodeType('j')}>Judgment Node</button>
        <button onClick={() => changeNodeType('p')}>Processing Node</button>
      </NodeToolbar>      
    </div>
  );
}

export default memo(gnpNode);