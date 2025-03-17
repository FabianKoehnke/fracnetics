import React, { memo } from 'react';
import {
  Handle,
  Position,  
  NodeToolbar,
} from '@xyflow/react';

function gnpNode({ data }) {
  return (
    <>
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
        <NodeToolbar
            isVisible={data.forceToolbarVisible || undefined}
            position={data.toolbarPosition}
            >
            <button>cut</button>
            <button>copy</button>
            <button>paste</button>
        </NodeToolbar>
        <div>{data?.label}</div>
    </>
  );
}
export default memo(gnpNode);