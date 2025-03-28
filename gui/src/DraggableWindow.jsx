import React, { useState, useEffect, useRef } from 'react';
import * as echarts from 'echarts';

const DraggableWindow = ({ dataFrame }) => {
  const [position, setPosition] = useState({ x: 100, y: 100 });
  const [isDragging, setIsDragging] = useState(false);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [size, setSize] = useState({ width: 300, height: 250 });
  const [isResizing, setIsResizing] = useState(false);
  const resizeHandleSize = 10;
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

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
        width: Math.max(100, e.clientX - position.x),
        height: Math.max(100, e.clientY - position.y),
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsResizing(false);
  };

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, isResizing, offset]);

  useEffect(() => {
    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    const generateX = (i, n) => Array.from({ length: n }, (_, index) => i + index);
    const updateChart = () => {
      if (chartInstance.current && dataFrame) {
        const option = {
          animation: true,
          tooltip: {},
          xAxis: {
            type: 'value',
            data: generateX(0,dataFrame[0].length) || [1,2,3,4,5,6,7],
            splitLine: { show: false },
          },
          yAxis: {
            type: 'value',
            splitLine: { show: false },
          },
          series: [
            {
              data: dataFrame[0] || [820, 932, 901, 934, 1290, 1330, 1320],
              type: 'line',
            },
          ],
        };

        chartInstance.current.setOption(option);
      }
    };

    updateChart();

    // Resize chart on window resize
    const handleResize = () => {
      if (chartInstance.current) {
        chartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      if (chartInstance.current) {
        chartInstance.current.dispose();
        chartInstance.current = null;
      }
      window.removeEventListener('resize', handleResize);
    };
  }, [dataFrame]);

  useEffect(() => {
    if (chartInstance.current) {
      chartInstance.current.resize();
    }
  }, [size]);

  return (
    <div
      style={{
        position: 'absolute',
        borderRadius: '20px',
        fontWeight: 'normal',
        fontFamily: 'Source Code Pro, monospace',
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
    >
      <div
        ref={chartRef}
        style={{
          width: '100%',
          height: `calc(100% - ${resizeHandleSize + 10}px)`,
        }}
      />
      <div
        className="resize-handle"
        style={{
          position: 'absolute',
          borderRadius: '3px',
          fontWeight: 'normal',
          fontFamily: 'Source Code Pro, monospace',
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

export default DraggableWindow;
