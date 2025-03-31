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

  function cumReturns(returns){
    let currentRet = 100;
    let profit = 0;
    const cumReturns = [];

    for(let i=0; i<returns.length;i++){
      profit = currentRet * returns[i];
      currentRet = currentRet + profit;
      cumReturns.push([i, currentRet]);
    }
    return cumReturns;
  }

  useEffect(() => {
    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    const updateChart = () => {
      if (chartInstance.current && dataFrame) {
        const cumrets = cumReturns(dataFrame[0]);
        const minValue = Math.round(Math.min(...cumrets.map(row => row[1])));
        const maxValue = Math.round(Math.max(...cumrets.map(row => row[1])));
        const option = {
          animation: true,
          tooltip: {},
          xAxis: {
            type: 'category',
            splitLine: { show: false },
          },
          yAxis: {
            type: 'value',
            splitLine: { show: false },
            min: minValue,
            max: maxValue
          },
          series: [
            {
              data: cumrets || [820, 932, 901, 934, 1290, 1330, 1320],
              type: 'line',
              lineStyle: {
                width: 2, 
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: '#f0f5f5' }, 
                    { offset: 1, color: '#979c9c' }  
                ])
            }
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