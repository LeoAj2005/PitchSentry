import os
import logging
from pathlib import Path
from ultralytics import YOLO
from config.settings import settings

logger = logging.getLogger(__name__)

class YOLOTrainer:
    """Handles the fine-tuning of the YOLO model on the Pitch_Sentry dataset."""
    
    def __init__(self, model_version: str = "yolo11n.pt"):
        # We start with a pre-trained 'nano' model for faster local training
        # Ultralytics will auto-download the base weights if not present
        logger.info(f"Initializing YOLO trainer with base model: {model_version}")
        self.model = YOLO(model_version)
        self.dataset_yaml = Path("datasets/yolo_format/dataset.yaml").absolute()

    def train(self, epochs: int = 50, batch_size: int = 16, imgsz: int = 640):
        """Executes the training loop."""
        if not self.dataset_yaml.exists():
            logger.error(f"Dataset config not found at {self.dataset_yaml}. Did you run dataset_prep.py?")
            return

        logger.info(f"Starting training on {self.dataset_yaml} for {epochs} epochs...")
        
        try:
            import torch
            # Hardware fallback logic: Use GPU if available, otherwise default to CPU
            actual_device = settings.DEVICE if torch.cuda.is_available() else "cpu"
            logger.info(f"Training will proceed on device: {actual_device}")

            # Training parameters optimized for football broadcast feeds
            results = self.model.train(
                data=str(self.dataset_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                device=actual_device,   # Use the validated device
                project="models",       
                name="pitch_sentry_v1", 
                exist_ok=True,          
                optimizer="auto",
                verbose=True
            )
            logger.info("Training completed successfully.")
            logger.info(f"Best weights saved to: models/pitch_sentry_v1/weights/best.pt")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    trainer = YOLOTrainer()
    # For testing on local machine, we use a tiny number of epochs and batch size
    trainer.train(epochs=3, batch_size=4)