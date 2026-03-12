import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_project_structure():
    """Scaffolds the modular directories for the Football Analytics AI System."""
    directories = [
        "config", "cloud", "data", "notebooks", "models",
        "vision", "tracking", "analytics", "physics", "api", "frontend"
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Verified directory: {dir_path}")
        
        # Add __init__.py to make them Python packages (except data, notebooks, frontend)
        if directory not in ["data", "notebooks", "frontend"]:
            init_file = os.path.join(dir_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    pass
                logging.info(f"Created module init: {init_file}")

if __name__ == "__main__":
    logging.info("Starting project scaffolding...")
    create_project_structure()
    logging.info("Project structure initialized successfully.")