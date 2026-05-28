import gradio as gr
import os
from pathlib import Path
import logging

from core.tracker import GroundedSAM2Tracker
from ui.components import create_ui
from utils.gaze_utils import GazeProcessor
import utils.TobiiPro3_PupilsNeon_to_csv as gaze_converter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the Multi-Video SAM2 Tracker"""
    
    # Initialize core components
    tracker = GroundedSAM2Tracker()
    gaze_processor = GazeProcessor()

    tracker.gaze_processor = gaze_processor

    # init_status = tracker.initialize_models()
    # logger.info(init_status)
    
    # Create the UI
    demo = create_ui(tracker, gaze_processor, gaze_converter)
    
    # Launch the application
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7870, 
        share=True,
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    main()
