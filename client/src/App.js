import React, { useEffect, useState } from 'react';

function App() {
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const [sensorDataList, setSensorDataList] = useState([]);
  const [drinkCount, setDrinkCount] = useState(0);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      setConnectionStatus('Connected');
      console.log('WebSocket connected');
    };

    ws.onclose = () => {
      setConnectionStatus('Disconnected');
      console.log('WebSocket disconnected');
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'SENSOR_DATA') {
        // Neue Sensordaten
        setSensorDataList((prev) => [msg.data, ...prev].slice(0, 100)); // z.B. max 100 Einträge
      } else if (msg.type === 'DRINK_DETECTED') {
        // Trinkvorgang erkannt
        setDrinkCount(msg.count);
        alert(`Trinkvorgang erkannt! Gesamt: ${msg.count}`);
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div className="App">
      <h1>Drink Monitoring</h1>
      <p>WebSocket-Status: {connectionStatus}</p>
      <p>Erkannte Trinkvorgänge: {drinkCount}</p>
      <h2>Letzte Sensordaten</h2>
      <div>
        {sensorDataList.map((entry, index) => (
          <div key={index} className="sensor-entry">
            <strong>{entry.timestamp}</strong> -
            ax: {entry.ax.toFixed(2)},
            ay: {entry.ay.toFixed(2)},
            az: {entry.az.toFixed(2)},
            gx: {entry.gx.toFixed(2)},
            gy: {entry.gy.toFixed(2)},
            gz: {entry.gz.toFixed(2)}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
