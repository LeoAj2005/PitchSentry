import numpy as np
import logging

logger = logging.getLogger(__name__)

class SpatialAnalyzer:
    """
    Computes spatial relationships and advanced metrics on the 2D pitch map.
    All inputs must be in real-world meters (x, y).
    """
    def __init__(self):
        # FIFA standard pitch dimensions
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        
        # Standard goal width is 7.32 meters. 
        # Placed perfectly in the center of the 68m width (Y-axis).
        self.goal_center = np.array([105.0, 34.0])
        self.post_left = np.array([105.0, 37.66])  # From shooter's perspective looking at goal
        self.post_right = np.array([105.0, 30.34])

    def calculate_distance(self, player_pos: tuple) -> float:
        """Calculates Euclidean distance from player to the center of the goal."""
        p_vec = np.array(player_pos)
        distance = np.linalg.norm(self.goal_center - p_vec)
        return float(distance)

    def calculate_shot_angle(self, shooter_pos: tuple) -> float:
        """
        Calculates the visible shot angle (in degrees) between the two goalposts.
        Uses the dot product of vectors from the shooter to each post.
        """
        s_vec = np.array(shooter_pos)
        
        # Vectors from shooter to posts
        v1 = self.post_left - s_vec
        v2 = self.post_right - s_vec
        
        # Magnitudes
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        # Cosine of the angle
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        # Clip to avoid floating point errors outside [-1, 1] for arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        
        return float(angle_deg)

    def analyze_shot_situation(self, shooter_pos: tuple, gk_pos: tuple = None) -> dict:
        """
        Compiles the spatial metrics for a given frame.
        """
        distance = self.calculate_distance(shooter_pos)
        shot_angle = self.calculate_shot_angle(shooter_pos)
        
        metrics = {
            "distance_meters": round(distance, 2),
            "shot_angle_degrees": round(shot_angle, 2),
            "is_inside_box": distance < 16.5 and (13.84 < shooter_pos[1] < 54.16)
        }
        
        # Basic GK coverage logic (can be expanded with defender cones)
        if gk_pos:
            gk_distance = self.calculate_distance(gk_pos)
            metrics["gk_distance_from_line"] = round(gk_distance, 2)
            
            # If GK is far off the line, they cut down the angle effectively
            metrics["gk_aggressive_positioning"] = gk_distance > 3.0
            
        return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    analyzer = SpatialAnalyzer()
    
    shooter_pos = (78.50, 35.10)
    gk_pos = (103.5, 34.5)
    
    logger.info("--- Analyzing Shooter Spatial Data ---")
    results = analyzer.analyze_shot_situation(shooter_pos, gk_pos)
    
    for key, value in results.items():
        logger.info(f"{key}: {value}")