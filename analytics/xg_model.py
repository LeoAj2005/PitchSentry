import os
import logging
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from pathlib import Path

logger = logging.getLogger(__name__)

class ExpectedGoalsModel:
    """
    XGBoost-based model to predict the probability of a shot resulting in a goal (xG).
    Uses spatial features derived from the 2D pitch map.
    """
    
    def __init__(self, model_dir: str = "models/xg_v1"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "xg_model.json"
        
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            max_depth=4,
            learning_rate=0.1,
            n_estimators=100,
            use_label_encoder=False
        )
        self.is_trained = False

    def _generate_synthetic_data(self, n_samples: int = 10000):
        """
        Generates realistic baseline training data for architectural testing.
        In production, replace this with a SQL query to your StatsBomb/Wyscout database.
        """
        logger.info(f"Generating {n_samples} synthetic shot records for training...")
        np.random.seed(42)
        
        # Features: distance (m), angle (deg), inside_box (1/0), gk_distance (m)
        distances = np.random.uniform(2.0, 35.0, n_samples)
        angles = np.random.uniform(5.0, 90.0, n_samples)
        inside_box = (distances < 16.5).astype(int)
        gk_distances = np.random.uniform(0.5, 6.0, n_samples)
        
        X = np.column_stack((distances, angles, inside_box, gk_distances))
        
        # Formulate a realistic probability equation for the labels
        # Closer distance and wider angle = higher probability
        base_prob = np.exp(-0.15 * distances) * (angles / 90.0) * 1.5
        # Add a slight boost if GK is far from the line
        base_prob += (gk_distances * 0.02)
        
        probabilities = np.clip(base_prob, 0.01, 0.95)
        
        # Generate binary outcomes (Goal = 1, Miss/Save = 0) based on probabilities
        y = np.random.binomial(1, probabilities)
        
        return X, y

    def train(self):
        """Trains the XGBoost model and saves the weights."""
        X, y = self._generate_synthetic_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info("Training XGBoost xG model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        preds_proba = self.model.predict_proba(X_test)[:, 1]
        loss = log_loss(y_test, preds_proba)
        auc = roc_auc_score(y_test, preds_proba)
        
        logger.info(f"Training Complete. Validation Log Loss: {loss:.4f}, AUC: {auc:.4f}")
        
        # Save artifact
        self.model.save_model(self.model_path)
        self.is_trained = True
        logger.info(f"Model saved to {self.model_path}")

    def load(self):
        """Loads pre-trained weights from disk."""
        if self.model_path.exists():
            self.model.load_model(self.model_path)
            self.is_trained = True
            logger.info(f"Loaded existing xG model from {self.model_path}")
        else:
            logger.warning("No saved model found. Please run train() first.")

    def predict_xg(self, spatial_features: dict) -> float:
        """
        Predicts Expected Goals (xG) from a dictionary of spatial features.
        """
        if not self.is_trained:
            self.load()
            if not self.is_trained:
                return 0.0
                
        # Must match the exact order of the synthetic data generation
        # [distance, angle, inside_box, gk_distance]
        features = np.array([[
            spatial_features.get("distance_meters", 20.0),
            spatial_features.get("shot_angle_degrees", 15.0),
            int(spatial_features.get("is_inside_box", 0)),
            spatial_features.get("gk_distance_from_line", 1.5)
        ]])
        
        xg_value = self.model.predict_proba(features)[0, 1]
        return float(xg_value)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    xg_engine = ExpectedGoalsModel()
    xg_engine.train()
    
    test_shot_features = {
        "distance_meters": 26.52,
        "shot_angle_degrees": 15.35,
        "is_inside_box": False,
        "gk_distance_from_line": 1.58
    }
    
    xg = xg_engine.predict_xg(test_shot_features)
    logger.info(f"--- Analysis Complete ---")
    logger.info(f"Shooter xG: {xg:.4f} ({xg*100:.1f}%)")