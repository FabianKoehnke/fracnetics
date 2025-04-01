import React, { useState } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import AddNodes from './GraphBuilder';
import '@xyflow/react/dist/style.css';
import './index.css';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div style={{ display: 'flex', width: '100vw', height: '100vh' }}>
      {/* Sidebar */}
      <div
        style={{
          width: sidebarOpen ? '250px' : '40px',
          background: '#090909',
          color: 'white',
          transition: 'width 0.3s ease',
          padding: '10px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <button
          style={{
            background: 'none',
            border: 'none',
            color: 'white',
            fontSize: '25px',
            textAlign: 'center',
            cursor: 'pointer',
          }}
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          ☰
        </button>
        {sidebarOpen && (
          <div>
            <p>Option 1</p>
            <p>Option 2</p>
            <p>Option 3</p>
          </div>
        )}
      </div>
      
      {/* Main content */}
      <div style={{ flexGrow: 1 }}>
        <ReactFlowProvider>
          <AddNodes />
        </ReactFlowProvider>
      </div>
    </div>
  );
}