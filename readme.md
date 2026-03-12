#  Pitch Sentry

A **Computer Vision + Machine Learning research prototype** that analyzes football shots from broadcast video and computes advanced analytics such as **Expected Goals (xG)**, shot angle, shot distance, goalkeeper coverage, and ball trajectory simulation.

This project combines **computer vision, spatial geometry, machine learning, and physics simulation** to build a system similar to modern football analytics platforms.

* * *

##  Project Overview

The system processes football broadcast footage and extracts meaningful analytics about shot quality.

### Pipeline

Video Input → Frame Extraction → Manual Pitch Calibration → Object Detection → Object Tracking → Pixel → Pitch Coordinate Mapping → Spatial Analytics → Expected Goals Model → Visualization / API Output

* * *

##  Features

### Computer Vision

• Player detection  
• Ball detection  
• Goalpost detection

### Tracking

• Multi-object player tracking  
• Ball trajectory tracking  
• Persistent player IDs

### Pitch Geometry

• Homography-based pitch mapping  
• Pixel → real-world coordinate conversion

### Shot Analytics

• Shot distance  
• Shot angle  
• Open goal area  
• Goalkeeper coverage

### Machine Learning

• Expected Goals (xG) prediction model  
• Defender block probability

### Physics Simulation

• Ball trajectory prediction  
• Required shot velocity estimation  
• Optimal shot direction

### Visualization

• Player position overlay  
• Ball trajectory visualization  
• Shot probability heatmaps

* * *

##  System Architecture

Video Frame  
↓  
YOLO Detection  
↓  
BoT-SORT Tracking  
↓  
Manual Pitch Calibration  
↓  
Homography Transformation  
↓  
Spatial Feature Extraction  
↓  
Machine Learning Model  
↓  
Analytics Output

* * *

##  Project Structure

football-analytics-ai/

config/  
data/  
notebooks/

models/  
 detection/  
 tracking/  
 xg\_model/

vision/  
 detect\_players.py  
 detect\_ball.py  
 manual\_calibrator.py  
 homography.py

tracking/  
 botsort\_tracker.py

analytics/  
 shot\_angle.py  
 shot\_distance.py  
 open\_goal\_area.py  
 block\_probability.py

physics/  
 trajectory.py

api/  
 main.py

frontend/

* * *

## ⚙ Installation

### Clone the repository
 
cd football-analytics-ai

### Create virtual environment

python -m venv venv

### Activate environment

Mac/Linux  
source venv/bin/activate

Windows  
venv\\Scripts\\activate

### Install dependencies

pip install -r requirements.txt

* * *

##  Manual Pitch Calibration

Because broadcast cameras vary between stadiums, the system requires **manual calibration once per video**.

### Workflow

1.  Load video frame.
    
2.  Click four pitch reference points:  
    • Left goalpost  
    • Right goalpost  
    • Left penalty box corner  
    • Right penalty box corner
    
3.  Homography matrix is computed.
    
4.  The matrix is reused for all frames in the clip.
    

Note: This ensures correct pixel → pitch coordinate mapping without training a complex pitch keypoint model.

* * *

##  Running the Pipeline

Example workflow:

Extract frames  
python scripts/extract\_frames.py

Run detection  
python vision/detect\_players.py

Run tracking  
python tracking/botsort\_tracker.py

Run analytics  
python analytics/shot\_analysis.py

* * *

##  Model Training (Cloud)

Models are trained using **Azure Machine Learning**.

Training workflow:

1.  Upload dataset to Azure Blob Storage.
    
2.  Train detection models on GPU.
    
3.  Export trained weights.
    
4.  Download model locally for inference.
    

* * *

##  Expected Goals (xG)

The xG model estimates the probability of a shot becoming a goal based on spatial features such as:

• Shot distance  
• Shot angle  
• Defender proximity  
• Goalkeeper coverage

* * *

##  Roadmap

✔ Session 1 – Project Setup  
✔ Session 2 – Data Pipeline  
✔ Session 3 – Object Detection  
✔ Session 4 – Player Tracking  
✔ Session 5 – Field Calibration  
✔ Session 6 – Shot Analytics

🔄 Session 7 – Expected Goals Model  
⏳ Session 8 – Ball Trajectory Simulation  
⏳ Session 9 – Defensive Analysis  
⏳ Session 10 – Visualization System  
⏳ Session 11 – API Server  
⏳ Session 12 – Full System Integration

* * *

## 🔬 Future Improvements

• Automatic pitch keypoint detection  
• Real-time analytics pipeline  
• Live broadcast integration  
• Advanced tactical analysis  
• Reinforcement learning shot optimization