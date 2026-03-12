import numpy as np
import logging

logger = logging.getLogger(__name__)

class DefensiveAnalyzer:
    """
    Computes defensive pressure, block probabilities, and pitch influence.
    Operates on 2D pitch coordinates (meters).
    """
    
    def __init__(self):
        self.goal_center = np.array([105.0, 34.0])
        self.post_left = np.array([105.0, 37.66])
        self.post_right = np.array([105.0, 30.34])

    def _point_in_triangle(self, pt: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> bool:
        """
        Uses vector cross products to determine if a point 'pt' is inside 
        the triangle formed by v1, v2, v3 (The Shot Cone).
        """
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def calculate_defensive_pressure(self, shooter_pos: tuple, defenders: list[tuple]) -> dict:
        """
        Calculates how much pressure the shooter is under and the probability 
        of the shot being blocked by the defense.
        """
        s_pos = np.array(shooter_pos)
        
        defenders_in_cone = 0
        max_block_prob = 0.0
        total_pressure = 0.0
        
        for def_pos in defenders:
            d_pos = np.array(def_pos)
            
            # 1. Pitch Influence / General Pressure
            # Modeled as a Gaussian decay. Pressure is 1.0 at 0m, decaying as distance increases.
            distance_to_shooter = np.linalg.norm(s_pos - d_pos)
            pressure = np.exp(-0.5 * distance_to_shooter) # standard decay
            total_pressure += pressure
            
            # 2. Shot Cone Intersection
            in_cone = self._point_in_triangle(d_pos, s_pos, self.post_left, self.post_right)
            
            if in_cone:
                defenders_in_cone += 1
                
                # Calculate specific block probability for this defender
                # A defender 9.15m away (standard wall distance) directly in the lane has high block chance
                
                # Calculate perpendicular distance to the center shot line
                shot_vector = self.goal_center - s_pos
                shot_vector_norm = shot_vector / np.linalg.norm(shot_vector)
                
                defender_vector = d_pos - s_pos
                projection_length = np.dot(defender_vector, shot_vector_norm)
                
                # If projection is negative, defender is behind the shooter
                if projection_length > 0:
                    projection_pt = s_pos + projection_length * shot_vector_norm
                    off_center_dist = np.linalg.norm(d_pos - projection_pt)
                    
                    # Block probability equation: 
                    # Decreases linearly with distance from shooter, exponentially with distance off-center
                    base_prob = max(0, 1.0 - (projection_length / 40.0)) 
                    alignment_penalty = np.exp(-2.0 * off_center_dist)
                    
                    block_prob = base_prob * alignment_penalty
                    max_block_prob = max(max_block_prob, block_prob)

        return {
            "defenders_in_shot_lane": defenders_in_cone,
            "max_block_probability": round(max_block_prob, 3),
            "total_defensive_pressure": round(total_pressure, 3),
            "shot_lane_openness": round(1.0 - max_block_prob, 3)
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    analyzer = DefensiveAnalyzer()
    
    shooter_pos = (78.5, 35.1)
    
    # Generic defensive wall
    defending_wall = [
        (87.5, 34.0), (87.5, 34.8), (87.5, 35.6), (87.5, 36.4), (95.0, 15.0) 
    ]
    
    logger.info("--- Analyzing Defensive Setup ---")
    results = analyzer.calculate_defensive_pressure(shooter_pos, defending_wall)
    
    for key, val in results.items():
        logger.info(f"{key}: {val}")