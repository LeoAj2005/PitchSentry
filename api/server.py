import logging
import cv2
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Import Pitch Sentry Modules
from vision.detector import PitchDetector
from vision.pitch_calibrator import PitchCalibrator
from analytics.spatial import SpatialAnalyzer
from analytics.xg_model import ExpectedGoalsModel
from analytics.defensive import DefensiveAnalyzer

logger = logging.getLogger(__name__)

app = FastAPI(title="Pitch Sentry AI API", version="1.0.0")

# Global variables for our AI models (loaded on startup to prevent latency)
models = {}

@app.on_event("startup")
async def load_models():
    """Loads massive AI models into RAM/VRAM once when the server starts."""
    logger.info("Booting up Pitch Sentry AI Engine...")
    models['detector'] = PitchDetector()
    models['spatial'] = SpatialAnalyzer()
    models['xg'] = ExpectedGoalsModel()
    models['defensive'] = DefensiveAnalyzer()
    models['calibrator'] = PitchCalibrator()
    logger.info("All AI models loaded and ready for inference.")

@app.get("/health")
async def health_check():
    """Simple endpoint to verify the server is running."""
    return {"status": "operational", "engine": "Pitch Sentry v1.0"}

@app.post("/analyze-frame")
async def analyze_frame(
    file: UploadFile = File(...),
    homography_filename: str = Form("H_messi_campnou.npy")
):
    """
    Master pipeline endpoint. 
    Ingests an image and outputs advanced tactical analytics.
    """
    # 1. Load the requested calibration matrix
    matrix_path = os.path.join("data", homography_filename)
    if not os.path.exists(matrix_path):
        raise HTTPException(status_code=400, detail=f"Calibration matrix not found: {matrix_path}")
    
    H_matrix = np.load(matrix_path)
    models['calibrator'].H = H_matrix

    # 2. Read the uploaded image into OpenCV format
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image payload.")

    # 3. Vision Pipeline: Detect all players and the ball
    detections = models['detector'].predict(frame)
    players = detections.get('players', [])
    balls = detections.get('ball', [])

    if not balls:
        return JSONResponse(status_code=422, content={"error": "No ball detected in frame. Cannot compute shot analytics."})
    if not players:
        return JSONResponse(status_code=422, content={"error": "No players detected in frame."})

    # For this snapshot, assume the ball is the first detected ball
    ball_bbox = balls[0]['bbox']
    ball_center = ((ball_bbox[0] + ball_bbox[2]) // 2, (ball_bbox[1] + ball_bbox[3]) // 2)

    # 4. Contextual Logic: Find the Shooter (Player closest to the ball)
    shooter = None
    min_dist = float('inf')
    defenders_pixels = []

    for p in players:
        p_bbox = p['bbox']
        # Bottom center of bounding box (feet)
        p_feet = ((p_bbox[0] + p_bbox[2]) // 2, p_bbox[3]) 
        
        # Distance to ball
        dist_to_ball = np.linalg.norm(np.array(p_feet) - np.array(ball_center))
        if dist_to_ball < min_dist:
            min_dist = dist_to_ball
            if shooter is not None:
                defenders_pixels.append(shooter) # Move previous closest to defenders
            shooter = p_feet
        else:
            defenders_pixels.append(p_feet)

    # 5. Mapping: Convert pixels to 2D Pitch Coordinates
    shooter_pitch = models['calibrator'].pixel_to_pitch(shooter)
    defenders_pitch = [models['calibrator'].pixel_to_pitch(d) for d in defenders_pixels]

    # 6. Analytics Pipeline
    spatial_data = models['spatial'].analyze_shot_situation(shooter_pitch)
    defensive_data = models['defensive'].calculate_defensive_pressure(shooter_pitch, defenders_pitch)
    
    # Merge spatial data with defensive data for the XGBoost model
    xg_features = {**spatial_data, **defensive_data}
    xg_value = models['xg'].predict_xg(xg_features)

    # 7. Construct Final API Payload
    response_payload = {
        "status": "success",
        "frame_resolution": {"width": frame.shape[1], "height": frame.shape[0]},
        "entities_detected": {"players": len(players), "balls": len(balls)},
        "analytics": {
            "expected_goals_xg": round(xg_value, 4),
            "spatial": spatial_data,
            "defensive": defensive_data
        },
        "coordinates_meters": {
            "shooter": {"x": round(shooter_pitch[0], 2), "y": round(shooter_pitch[1], 2)}
        }
    }

    return response_payload

if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)