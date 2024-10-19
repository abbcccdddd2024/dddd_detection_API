from .detection import getDetections
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import json
import time
import threading

app = FastAPI()

json_file_path = "detections.json"
json_data_cache = None  # Cache to hold the JSON data

# Background task to update the cache continuously
def update_json_cache():
    global json_data_cache
    while True:
        try:
            with open(json_file_path, 'r') as file:
                json_data_cache = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            json_data_cache = None

        # Check for updates every 0.5 seconds
        time.sleep(0.5)

# Start the background thread
threading.Thread(target=update_json_cache, daemon=True).start()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/viewData")
def read_view_data():
    return getDetections()

@app.get("/getViewData")
async def get_view_data():
    with open('detections.json', 'r') as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/detections")
async def get_detections():
    global json_data_cache
    if json_data_cache is None:
        return JSONResponse(content={"error": "JSON file not found or not ready."}, status_code=404)
    return JSONResponse(content=json_data_cache)

