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
  MarkerType
} from '@xyflow/react';

import '@xyflow/react/dist/style.css';
import './index.css';
import gnpNode from "./gnpNode";
import FloatingEdge from './FloatingEdge';

const nodeTypes = { gnpNode };

const initialNodes = [
  {
    id: "0",
    type: "gnpNode",
    data: { label: `SN ${0}` },
    position: { x: 0, y: 0 },
    className: 'startNode',
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
    [setEdges],
  );

  const edgeTypes = {
    floating: FloatingEdge,
  };

  const defaultEdgeOptions = {
    type: 'floating',
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: '#b1b1b7',
    },
  };

  
  const onConnectEnd = useCallback(
    (event, connectionState) => {
      
      const outgoingEdges = edges.filter((edge) => edge.source === connectionState?.fromNode?.id);
      if (!connectionState.isValid && (outgoingEdges.length < 1 || connectionState?.fromNode?.data.label.slice(0,2) == "JN")) {
        const id = getId();
        const { clientX, clientY } = 'changedTouches' in event ? event.changedTouches[0] : event;
        const newNode = {
          id,
          type: 'gnpNode',
          position: screenToFlowPosition({
            x: clientX,
            y: clientY,
          }),
          data: { label: `N ${id}` },
          origin: [0.5, 0.0],
          style: { background: "#ffffff66" }
        };

        setNodes((nds) => nds.concat(newNode));
        setEdges((eds) =>
          eds.concat({
            id,
            source: connectionState.fromNode.id,
            target: id,
            animated: true
          }),
        );
      }
    },
    [screenToFlowPosition, edges, setNodes, setEdges], 
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
        edgeTypes={edgeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
      >
        <Background />
        <Controls />
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