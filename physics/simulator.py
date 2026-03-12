import numpy as np
import logging
from scipy.signal import savgol_filter
from typing import List, Tuple

logger = logging.getLogger(__name__)

class BallPhysics:
    """
    Handles trajectory smoothing and kinematic calculations for the ball.
    Requires input points mapped to the 2D pitch in meters (from Session 5).
    """
    
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.dt = 1.0 / self.fps  # Delta time between frames

    def smooth_trajectory(self, points: np.ndarray, window_length: int = 7, polyorder: int = 2) -> np.ndarray:
        """
        Applies a Savitzky-Golay filter to smooth noisy trajectory data.
        """
        if len(points) < window_length:
            logger.warning("Not enough points to smooth trajectory. Returning raw points.")
            return points

        # Separate X and Y coordinates
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # Apply filter independently to X and Y
        x_smooth = savgol_filter(x_coords, window_length, polyorder)
        y_smooth = savgol_filter(y_coords, window_length, polyorder)

        # Recombine
        smoothed_points = np.column_stack((x_smooth, y_smooth))
        return smoothed_points

    def calculate_velocities(self, smoothed_points: np.ndarray) -> np.ndarray:
        """
        Calculates the instantaneous velocity (in km/h) between consecutive frames.
        """
        if len(smoothed_points) < 2:
            return np.array([])

        velocities_kmh = []
        
        # Iterate through points to find distance traveled per frame
        for i in range(1, len(smoothed_points)):
            p1 = smoothed_points[i - 1]
            p2 = smoothed_points[i]
            
            # Distance in meters
            distance_m = np.linalg.norm(p2 - p1)
            
            # Speed = Distance / Time (m/s)
            speed_ms = distance_m / self.dt
            
            # Convert to km/h (1 m/s = 3.6 km/h)
            speed_kmh = speed_ms * 3.6
            velocities_kmh.append(speed_kmh)
            
        return np.array(velocities_kmh)

    def analyze_shot(self, raw_pitch_points: List[Tuple[float, float]]) -> dict:
        """
        Runs the full physics pipeline on a sequence of ball coordinates.
        """
        points_array = np.array(raw_pitch_points)
        
        smoothed_path = self.smooth_trajectory(points_array)
        velocities = self.calculate_velocities(smoothed_path)
        
        if len(velocities) == 0:
            return {"error": "Insufficient trajectory data"}

        top_speed = np.max(velocities)
        avg_speed = np.mean(velocities)
        
        # Calculate overall distance of the shot (first point to last point)
        total_distance = np.linalg.norm(smoothed_path[-1] - smoothed_path[0])

        return {
            "top_speed_kmh": round(top_speed, 2),
            "average_speed_kmh": round(avg_speed, 2),
            "total_distance_meters": round(total_distance, 2),
            "trajectory_points": smoothed_path.tolist()
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Assuming standard broadcast is 30 FPS
    physics_engine = BallPhysics(fps=30)
    
    # --- SIMULATE A NOISY FREEKICK TRAJECTORY ---
    # Imagine Messi shooting from (78.5, 35.1) to the goal at (105.0, 36.0) over 1 second (30 frames)
    # We generate a straight line, then inject random noise to simulate bounding box jitter
    logger.info("Generating synthetic noisy ball tracking data...")
    num_frames = 30
    x_true = np.linspace(78.5, 105.0, num_frames)
    y_true = np.linspace(35.1, 36.0, num_frames)
    
    # Add +/- 0.5 meters of random jitter (which happens constantly in 2D tracking)
    np.random.seed(42)
    x_noisy = x_true + np.random.normal(0, 0.5, num_frames)
    y_noisy = y_true + np.random.normal(0, 0.5, num_frames)
    
    noisy_trajectory = list(zip(x_noisy, y_noisy))
    
    logger.info("Running Physics Analysis pipeline...")
    results = physics_engine.analyze_shot(noisy_trajectory)
    
    logger.info("--- Physics Simulation Results ---")
    logger.info(f"Shot Distance Traveled: {results['total_distance_meters']} meters")
    logger.info(f"Average Speed: {results['average_speed_kmh']} km/h")
    logger.info(f"Top Speed (Peak Velocity): {results['top_speed_kmh']} km/h")