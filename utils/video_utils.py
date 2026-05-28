import cv2
import os
import glob
from pathlib import Path
import logging
from PIL import Image
import gradio as gr
import numpy as np

logger = logging.getLogger(__name__)


def create_video_from_images(images_dir, output_path, fps=25, image_pattern="*.jpg"):
    """
    Create video from directory of images
    
    Args:
        images_dir: Directory containing image frames
        output_path: Output video file path
        fps: Frames per second for output video
        image_pattern: Pattern to match image files (e.g., "*.jpg", "annotated_frame_*.jpg")
    """
    try:
        # Get all image files matching pattern
        if image_pattern == "*.jpg":
            # For annotated frames, use specific pattern
            image_files = glob.glob(os.path.join(images_dir, "annotated_frame_*.jpg"))
            if not image_files:
                # Fallback to all jpg files
                image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
        else:
            image_files = glob.glob(os.path.join(images_dir, image_pattern))
        
        if not image_files:
            logger.warning(f"No images found in {images_dir} with pattern {image_pattern}")
            return False
        
        # Sort files naturally (handle numeric sorting)
        def natural_sort_key(path):
            try:
                # Extract frame number from filename for proper sorting
                filename = Path(path).stem
                if "frame_" in filename:
                    frame_num = filename.split("frame_")[-1]
                    return int(frame_num)
                else:
                    return int(''.join(filter(str.isdigit, filename)))
            except (ValueError, IndexError):
                return 0
        
        image_files.sort(key=natural_sort_key)
        
        # Read first image to get dimensions
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            logger.error(f"Could not read first image: {image_files[0]}")
            return False
        
        height, width, layers = first_image.shape
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            logger.error(f"Could not open video writer for {output_path}")
            return False
        
        # Write each frame
        for i, image_file in enumerate(image_files):
            frame = cv2.imread(image_file)
            if frame is not None:
                # Ensure frame has correct dimensions
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                video_writer.write(frame)
            else:
                logger.warning(f"Could not read frame: {image_file}")
        
        # Release everything
        video_writer.release()
        
        # Verify output file was created
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Successfully created video: {output_path} from {len(image_files)} frames")
            return True
        else:
            logger.error(f"Failed to create video: {output_path}")
            return False
        
    except Exception as e:
        logger.error(f"Error creating video from images: {e}")
        return False

# Define click handler function
def handle_image_click(evt: gr.SelectData, image, correction_mode, pos_points, neg_points, current_frame):
    """Handle click events on the image to add points"""
    try:
        # Get click coordinates
        x, y = evt.index
        
        # Record the point based on correction mode

        if correction_mode == "New Object":
            # For new object mode, always add as positive point and clear existing points
            new_pos_points = [(x, y)]
            new_neg_points = []
            info_msg = f"Frame {current_frame}: 1 positive, 0 negative (NEW OBJECT - add more points as needed)"
        elif correction_mode == "Add (+)":
            new_pos_points = pos_points + [(x, y)]
            new_neg_points = neg_points
            info_msg = f"Frame {current_frame}: {len(new_pos_points)} positive, {len(new_neg_points)} negative"
        elif correction_mode == "Remove (-)":
            new_pos_points = pos_points
            new_neg_points = neg_points + [(x, y)]
            info_msg = f"Frame {current_frame}: {len(new_pos_points)} positive, {len(new_neg_points)} negative"
        
        # Draw points on the image for visualization
        if image is not None:
            # Convert PIL Image to numpy array while preserving RGB format
            image_np = np.array(image)
            
            # Ensure we're working with RGB format (not BGR)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                # Image is already in RGB format from PIL, no conversion needed
                pass
            
            # Draw positive points with + symbol
            for px, py in new_pos_points:
                # Draw white circle background
                cv2.circle(image_np, (int(px), int(py)), 12, (255, 255, 255), -1)
                # Draw green circle border
                cv2.circle(image_np, (int(px), int(py)), 12, (0, 255, 0), 2)
                
                # Draw + symbol
                # Horizontal line
                cv2.line(image_np, (int(px-6), int(py)), (int(px+6), int(py)), (0, 255, 0), 2)
                # Vertical line  
                cv2.line(image_np, (int(px), int(py-6)), (int(px), int(py+6)), (0, 255, 0), 2)
                
            # Draw negative points with - symbol
            for nx, ny in new_neg_points:
                # Draw white circle background
                cv2.circle(image_np, (int(nx), int(ny)), 12, (255, 255, 255), -1)
                # Draw red circle border
                cv2.circle(image_np, (int(nx), int(ny)), 12, (255, 0, 0), 2)
                
                # Draw - symbol (horizontal line only)
                cv2.line(image_np, (int(nx-6), int(ny)), (int(nx+6), int(ny)), (255, 0, 0), 2)

            
            # Return the updated image with points (keep as RGB numpy array)
            return image_np, new_pos_points, new_neg_points, info_msg
        
        # Return the updated lists and info message without image
        return gr.update(), new_pos_points, new_neg_points, info_msg

    except Exception as e:
        logger.error(f"Error handling click point: {e}")
        return gr.update(), pos_points, neg_points, f"Error: {str(e)}"
