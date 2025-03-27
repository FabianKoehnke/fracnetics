import React, { useState, useEffect } from 'react';

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
      Draggable Window
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
