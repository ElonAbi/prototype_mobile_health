import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
from ble_service import BLEService

app = FastAPI()

active_connections: List[WebSocket] = []

ble_service = BLEService(model_path='drink_detection_model.pkl')

async def broadcast_message(message: dict):
    disconnected = []
    for conn in active_connections:
        try:
            await conn.send_json(message)
        except WebSocketDisconnect:
            disconnected.append(conn)
        except:
            disconnected.append(conn)
    for conn in disconnected:
        active_connections.remove(conn)

def on_new_data(sensor_dict):
    asyncio.create_task(broadcast_message({
        "type": "SENSOR_DATA",
        "data": sensor_dict
    }))

def on_drink_detected(drink_count):
    asyncio.create_task(broadcast_message({
        "type": "DRINK_DETECTED",
        "count": drink_count
    }))

ble_service.on_data_callback = on_new_data
ble_service.on_drink_callback = on_drink_detected

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()  # Falls der Client was sendet
            print("WebSocket received:", data)
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print("WebSocket disconnected")

@app.on_event("startup")
async def startup_event():
    connected = await ble_service.connect()
    if not connected:
        print("Konnte nicht mit der Smartwatch verbinden.")

@app.on_event("shutdown")
async def shutdown_event():
    await ble_service.disconnect()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
