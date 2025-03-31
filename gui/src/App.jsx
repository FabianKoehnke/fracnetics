/*
Main file of the App
*/

// Imports
import React, { useCallback, useRef, useEffect, useState } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import AddNodes from './GraphBuilder'
import '@xyflow/react/dist/style.css';
import './index.css';

export default function App() {
  return (
    <div style={{ display: 'flex', width: '100vw', height: '100vh' }}>
      <ReactFlowProvider>
        <AddNodes />
      </ReactFlowProvider>
    </div>
  );
}
