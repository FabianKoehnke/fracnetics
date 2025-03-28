import React, { useCallback, useRef, useEffect, useState } from 'react';
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
  MarkerType,
} from '@xyflow/react';

import '@xyflow/react/dist/style.css';
import gnpNode from './gnpNode';
import FloatingEdge from './FloatingEdge';
import { handleGraphData } from './graphDataHandler';
import DraggableWindow from './DraggableWindow';
import './index.css';

const nodeTypes = { gnpNode };

const initialNodes = [
  {
    id: '0',
    type: 'gnpNode',
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
  const { screenToFlowPosition, getSelectedNodes, setNodes: setReactFlowNodes, setEdges: setReactFlowEdges } = useReactFlow();

  const [dataFrame, setDataFrame] = useState(null); // Zustand für den dataFrame

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
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
      const outgoingEdges = edges.filter(
        (edge) => edge.source === connectionState?.fromNode?.id
      );
      if (
        !connectionState.isValid &&
        (outgoingEdges.length < 1 || connectionState?.fromNode?.data.label.slice(0, 2) === 'JN')
      ) {
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
          style: { background: '#ffffff66' },
        };

        setNodes((nds) => nds.concat(newNode));
        setEdges((eds) =>
          eds.concat({
            id,
            source: connectionState.fromNode.id,
            target: id,
            animated: true,
          })
        );
      }
    },
    [screenToFlowPosition, edges, setNodes, setEdges]
  );

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        const selectedNodes = getSelectedNodes();
        if (selectedNodes.length > 0) {
          const nodeIdsToDelete = selectedNodes.map((node) => node.id);
          setNodes((nds) => nds.filter((node) => !nodeIdsToDelete.includes(node.id)));
          setEdges((eds) =>
            eds.filter(
              (edge) =>
                !nodeIdsToDelete.includes(edge.source) &&
                !nodeIdsToDelete.includes(edge.target)
            )
          );
          setReactFlowNodes((nds) => nds.filter((node) => !nodeIdsToDelete.includes(node.id)));
          setReactFlowEdges((eds) =>
            eds.filter(
              (edge) =>
                !nodeIdsToDelete.includes(edge.source) &&
                !nodeIdsToDelete.includes(edge.target)
            )
          );
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [getSelectedNodes, setNodes, setEdges, setReactFlowNodes, setReactFlowEdges]);

  // Hier aktualisieren wir den dataFrame, wenn sich nodes oder edges ändern
  useEffect(() => {
    const newDataFrame = handleGraphData(nodes, edges);
    setDataFrame(newDataFrame); // Speichern des neuen dataFrames
  }, [nodes, edges]);

  return (
    <div style={{ width: '100%', height: '100%' }} className="wrapper" ref={reactFlowWrapper}>
      <ReactFlow
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
        <Controls className='controler' />
        <Background color="#ddd" variant="" />
      </ReactFlow>

      {/* Hier übergeben wir den dataFrame als Prop an die DraggableWindow-Komponente */}
      <DraggableWindow dataFrame={dataFrame} />
    </div>
  );
};

export default function App() {
  return (
    <div style={{ display: 'flex', width: '100vw', height: '100vh' }}>
      <ReactFlowProvider>
        <AddNodeOnEdgeDrop />
      </ReactFlowProvider>
    </div>
  );
}
