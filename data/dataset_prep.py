import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class YOLODatasetFormatter:
    """Generates the required YOLO architecture and configuration file."""
    
    def __init__(self, base_dir: str = "datasets/yolo_format"):
        self.base_dir = Path(base_dir)

    def setup_directories(self):
        """Creates the train/val split directories for images and labels."""
        splits = ['train', 'val', 'test']
        subdirs = ['images', 'labels']
        
        for split in splits:
            for subdir in subdirs:
                dir_path = self.base_dir / subdir / split
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")

    def generate_yaml(self):
        """Generates the dataset.yaml file required by Ultralytics."""
        yaml_path = self.base_dir / "dataset.yaml"
        
        # We need to detect players, the ball, and goalposts.
        data = {
            'path': str(self.base_dir.absolute()), 
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {
                0: 'player',
                1: 'ball',
                2: 'goalpost'
            }
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Generated YOLO config: {yaml_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    formatter = YOLODatasetFormatter()
    formatter.setup_directories()
    formatter.generate_yaml()