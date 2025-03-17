import React, { useCallback, useRef } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  useReactFlow,
  ReactFlowProvider,
} from '@xyflow/react';
 
import '@xyflow/react/dist/style.css';
import gnpNode from "./gnpNode"

const nodeTypes = {gnpNode};

const initialNodes = [
  {
    id: "0",
    type: 'gnpNode',
    data: {label: `N ${0}`},
    position: { x: 0, y: 50 },
  },
];

let id = 1;
const getId = () => `${id++}`;
const nodeOrigin = [0.5, 0];
const AddNodeOnEdgeDrop = () => {
  
  const reactFlowWrapper = useRef(null);
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const { screenToFlowPosition } = useReactFlow();
  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [],
  );
 
  const onConnectEnd = useCallback(
    (event, connectionState) => {
      // when a connection is dropped on the pane it's not valid
      if (!connectionState.isValid) {
        // we need to remove the wrapper bounds, in order to get the correct position
        const id = getId();
        const { clientX, clientY } =
          'changedTouches' in event ? event.changedTouches[0] : event;
        const newNode = {
          id,
          type: 'gnpNode',
          position: screenToFlowPosition({
            x: clientX,
            y: clientY,
          }),
          data: { label: `N ${id}` },
          origin: [0.5, 0.0],
        };
 
        setNodes((nds) => nds.concat(newNode));
        setEdges((eds) =>
          eds.concat({ id, source: connectionState.fromNode.id, target: id , animated: true}),
        );
      }
    },
    [screenToFlowPosition],
  );
 
  return (
    <div style={{ width: '100vw', height: '100vh' }} className="wrapper" ref={reactFlowWrapper}>
      <ReactFlow
        style={{ backgroundColor: "#F79FB" }}
        nodes={nodes}
        nodeTypes={nodeTypes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onConnectEnd={onConnectEnd}
        fitView
        fitViewOptions={{ padding: 2 }}
        nodeOrigin={nodeOrigin}
      >
      <Background />
      <Controls />
        <MiniMap />
        <Background color="#ddd" variant="dots" gap={12} size={1} />
      </ReactFlow>
    </div>
  );
};
 
export default function App() {
  
  return (
    <div>
      <ReactFlowProvider>
        <AddNodeOnEdgeDrop />
      </ReactFlowProvider> 
    </div>

  );
}