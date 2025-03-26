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
        (outgoingEdges.length < 1 ||
          connectionState?.fromNode?.data.label.slice(0, 2) === 'JN')
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

  useEffect(() => {
    handleGraphData(nodes, edges);
  }, [nodes, edges]);

  return (
    <div
      style={{ width: '100%', height: '100%' }}
      className="wrapper"
      ref={reactFlowWrapper}
    >
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
        <Controls className='controler'/>
        <Background color="#ddd" variant="" />
      </ReactFlow>
    </div>
  );
};

const DraggableWindow = () => {
  const [position, setPosition] = useState({ x: 100, y: 100 });
  const [isDragging, setIsDragging] = useState(false);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [size, setSize] = useState({ width: 300, height: 200 });
  const [isResizing, setIsResizing] = useState(false);
  const resizeHandleSize = 10;

  const handleMouseDown = (e) => {
    if (e.target.classList.contains('resize-handle')) {
      setIsResizing(true);
      setOffset({
        x: e.clientX - (position.x + size.width),
        y: e.clientY - (position.y + size.height),
      });
    } else {
      setIsDragging(true);
      setOffset({
        x: e.clientX - position.x,
        y: e.clientY - position.y,
      });
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - offset.x,
        y: e.clientY - offset.y,
      });
    } else if (isResizing) {
      setSize({
        width: Math.max(100, e.clientX - position.x - offset.x),
        height: Math.max(100, e.clientY - position.y - offset.y),
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsResizing(false);
  };

  return (
    <div
      style={{
        position: 'absolute',
        borderRadius: '20px',
        left: position.x,
        top: position.y,
        width: size.width,
        height: size.height,
        border: '1px solid black',
        backgroundColor: '#090909',
        padding: '10px',
        zIndex: 1000,
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      Draggable Window
      <div
        className="resize-handle"
        style={{
          position: 'absolute',
          borderRadius: '3px',
          right: 0,
          bottom: 0,
          width: resizeHandleSize,
          height: resizeHandleSize,
          backgroundColor: 'white',
          cursor: 'se-resize',
        }}
      />
    </div>
  );
};

export default function App() {
  return (
    <div style={{ display: 'flex', width: '100vw', height: '100vh' }}>
        <ReactFlowProvider>
          <AddNodeOnEdgeDrop />
        </ReactFlowProvider>
      
        <DraggableWindow />
    </div>
  );
}