import React, { memo, useCallback, useState, useEffect } from 'react';
import { Handle, Position, NodeToolbar, useReactFlow, useEdges } from '@xyflow/react';
import "./index.css"

function gnpNode({ id, data }) {
  const { setNodes, getEdges } = useReactFlow();
  const edges = getEdges();
  const outgoingEdgesCount = edges.filter(edge => edge.source === id).length;
  const [inputValue, setInputValue] = useState(data.nodeValues ? data.nodeValues.join(',') : '');
  const [inputFunction, setfunctionInput] = useState(data.nodeFunction ? data.nodeFunction : '');
  const [inputDistribution, setDistribution] = useState(data.distribution ? data.distribution : '');

  useEffect(() => {
    if (data.nodeValues) {
      setInputValue(data.nodeValues.join(','));
      setfunctionInput(data.nodeFunction);
      setDistribution(data.distribution);
    } else {
      setInputValue('');
      setfunctionInput('');
      setDistribution('');
    }
  }, [data.nodeValues, data.nodeFunction, data.distribution]);

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

  const handleInputValues = (event) => {
    const newValue = event.target.value;
    setInputValue(newValue);
  };

  const handleInputFunction = (event) => {
    const newValue = event.target.value;
    setfunctionInput(newValue);
  };

  const handleDistributionChange = (distribution) => {
    setDistribution(distribution);
  };

  useEffect(() => {
    setNodes((nodes) =>
      nodes.map((node) => {
        if (node.id === id) {
          let inputValues = inputValue.split(',').map(v => v.trim()).filter(v => v !== '');
          let inputFunctions = inputFunction;
          let inputDistributions = inputDistribution;

          if (data.label.slice(0, 2) === "PN") {
            if (inputValues.length <= 1) {
              inputValues = inputValues;
            } else {
              inputValues = "";
            }
          } else if (data.label.slice(0, 2) === "JN") {
            if (inputValues.length === outgoingEdgesCount + 1) {
              inputValues = inputValues;
            } else {
              inputValues = "";
            }
          }

          return {
            ...node,
            data: {
              ...node.data,
              nodeValues: inputValues,
              nodeFunction: inputFunctions,
              distribution: inputDistributions,
            },
          };
        }
        return node;
      })
    );
  }, [inputValue, inputFunction, inputDistribution, id, setNodes, outgoingEdgesCount]);

  if (data.label.slice(0, 1) === "N") {
    return (
      <div className={`gnp-node-outer ${data.style}`} style={{ backgroundColor: data?.color }}>
        <div className="gnp-node-inner" style={{ backgroundColor: data?.color }}>
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
  } else if (data.label.slice(0, 2) === "PN") {
    return (
      <div className={`gnp-node-outer ${data.style}`} style={{ backgroundColor: data?.color }}>
        <div className="gnp-node-inner" style={{ backgroundColor: data?.color }}>
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
            className='inputValues'
            size="14"
            type="text"
            value={inputValue}
            onChange={handleInputValues}
            placeholder="Enter y-value"
          />
        </NodeToolbar>
        <NodeToolbar className="toolbar" isVisible={data.forceToolbarVisible || undefined} position={Position.Right}>
          <div className="dropdown">
            <button>{inputDistribution || "Distribution"}</button>
            <div className="dropdown-content">
              <button onClick={() => handleDistributionChange("Uniform")}>Uniform</button>
              <button onClick={() => handleDistributionChange("Normal")}>Normal</button>
            </div>
          </div>
        </NodeToolbar>
      </div>
    );
  } else if (data.label.slice(0, 2) === "JN") {
    return (
      <div
        className={`gnp-node-outer ${data.style}`}
        style={{ backgroundColor: data?.color }}
      >
        <div
          className={`gnp-node-inner ${data.style}`}
          style={{ backgroundColor: data?.color }}
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
            className='inputValues'
            size="35"
            type="text"
            value={inputValue}
            onChange={handleInputValues}
            placeholder="Enter comma-separated boundaries"
          />
        </NodeToolbar>
        <NodeToolbar className="toolbar" isVisible={data.forceToolbarVisible || undefined} position={Position.Left}>
          <input
            className='inputValues'
            size="14"
            type="text"
            value={inputFunction}
            onChange={handleInputFunction}
            placeholder="Enter function"
          />
        </NodeToolbar>
        <NodeToolbar className="toolbar" isVisible={data.forceToolbarVisible || undefined} position={Position.Right}>
          <div className="dropdown">
            <button>{inputDistribution || "Distribution"}</button>
            <div className="dropdown-content">
              <button onClick={() => handleDistributionChange("Uniform")}>Uniform</button>
              <button onClick={() => handleDistributionChange("Normal")}>Normal</button>
            </div>
          </div>
        </NodeToolbar>
      </div>
    );
  } else if (data.label.slice(0, 2) === "SN") {
    return (
      <div className={`gnp-node-outer ${data.style}`} style={{ backgroundColor: data?.color }}>
        <div className="gnp-node-inner" style={{ backgroundColor: data?.color }}>
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