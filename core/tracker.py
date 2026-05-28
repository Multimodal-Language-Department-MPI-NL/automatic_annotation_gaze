import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import tempfile
import shutil
import json
from typing import List, Dict, Tuple, Optional
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils.video_utils import create_video_from_images
from utils.gaze_utils import GazeProcessor

from utils.elan_exporter import ELANExporter, merge_consecutive_frames

logger = logging.getLogger(__name__)

class GroundedSAM2Tracker:
    def __init__(self):
        self.video_predictor = None
        self.image_predictor = None
        self.grounding_model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_video_sessions = {}
        self.temp_base_dir = None
        self.gaze_processor = GazeProcessor()  # Integrated gaze processor

        self.current_object_id = 0
        self.initial_prompts = []   # list of dicts as you already use elsewhere
        self.saved_points = {}      # {frame_idx: {obj_id: {"positive":[], "negative":[]}}}
        self.negative_threshold = 0.5  # Higher = stronger negative point impact

    def initialize_models(self):
        """Initialize SAM2 and Grounding DINO models"""
        try:
            # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            
            self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
            sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
            self.image_predictor = SAM2ImagePredictor(sam2_image_model)
            
            # Grounding DINO model initialization (only for text prompts)
            model_id = "IDEA-Research/grounding-dino-base"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            
            logger.info("✅ Models loaded successfully!")
            # return "✅ Models loaded successfully!"
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            # return f"❌ Error loading models: {str(e)}"
    
    def process_video_input(self, video_file, video_dir_path, gaze_loaded):
        """Video processing that also loads gaze data if provided"""
        try:
            # Clean up previous sessions
            if self.temp_base_dir and os.path.exists(self.temp_base_dir):
                shutil.rmtree(self.temp_base_dir)
            
            self.current_video_sessions = {}
            
            # Create base temp directory
            self.temp_base_dir = tempfile.mkdtemp(prefix="sam2_tracker_")
            
            video_files = []
            
            if video_file is not None:
                video_path = os.path.join(self.temp_base_dir, "uploaded_video" + Path(video_file.name).suffix)
                shutil.copy2(video_file.name, video_path)
                video_files = [video_path]
                
            elif video_dir_path and os.path.exists(video_dir_path):
                    video_files = self.get_video_files(video_dir_path)
            else:
                return "❌ Please provide either a video file or valid directory path", None, "", []
            
            if not video_files:
                return "❌ No valid video files found", None, "", []
            
            # Process each video
            processed_videos = []
            
            for i, video_path in enumerate(video_files):
                try:
                    video_name = Path(video_path).stem
                    video_frames_dir = os.path.join(self.temp_base_dir, f"video_{i:03d}_{video_name}")
                    os.makedirs(video_frames_dir, exist_ok=True)
                    
                    metadata = self.get_video_metadata(video_path)
                    if not metadata:
                        logger.warning(f"Could not get metadata for {video_path}")
                        continue
                    
                    frame_names = self.extract_frames_from_video(video_path, video_frames_dir)
                    
                    if not frame_names:
                        logger.warning(f"No frames extracted from {video_path}")
                        continue
                    
                    # Store video session info with gaze integration
                    video_session = {
                        'frames_dir': video_frames_dir,
                        'frame_names': frame_names,
                        'video_path': video_path,
                        'video_name': video_name,
                        'inference_state': None,
                        'segments': {},
                        'tracked_objects': {},  
                        'has_gaze_data': gaze_loaded,
                        'confidence_data': {
                            "frames": [],
                            "iou_predictions": [],
                            "occlusion_predictions": []
                        },
                        'object_confidence_data': {}  # Per-object confidence: {obj_id: {"frames": [], "iou_predictions": [], "occlusion_predictions": []}}
            }
                    
                    video_key = f"video_{i:03d}"
                    self.current_video_sessions[video_key] = video_session
                    
                    processed_videos.append(video_name)
                    logger.info(f"Processed video {video_name}: {len(frame_names)} frames")
                    
                except Exception as e:
                    logger.error(f"Error processing video {video_path}: {e}")
                    continue
            
            if not self.current_video_sessions:
                return "❌ No videos were successfully processed", None, "", []
            
            
            gaze_status = " 👁️ with gaze overlay" if gaze_loaded else ""
            status_msg = f"✅ Successfully processed {len(processed_videos)} video(s){gaze_status}\n✅ Ready for object detection"
            
            return status_msg
            
        except Exception as e:
            logger.error(f"Error processing video input: {e}")
            return f"❌ Error processing video: {str(e)}", None, "", []

    def _add_tracked_object(self, video_key, obj_id, obj_type, frame_idx, 
                           label="object"):
        """Add object to tracking with all its data in one place"""
        video_session = self.current_video_sessions[video_key]
        if 'tracked_objects' not in video_session:
            video_session['tracked_objects'] = {}
            
        video_session['tracked_objects'][obj_id] = {
            'obj_id': obj_id,
            'label': label,
            'type': obj_type,  # 'box', 'visual_points', etc.
            'created_frame': frame_idx,
            'correction_points': {},     # {frame_idx: {"positive": [], "negative": []}}
            'confidence_data': {"frames": [], "iou_predictions": [], "occlusion_predictions": []}
        }
    
    def get_tracked_frame_data(self, video_key, frame_idx, specific_obj_id=None):
        """Get tracking data for a specific frame, showing all objects but highlighting the selected one"""
        
        try:
            if video_key not in self.current_video_sessions:
                logger.error(f"Invalid video key: {video_key}")
                return None, "❌ Invalid video selection"
            
            video_session = self.current_video_sessions[video_key]
            
            if frame_idx >= len(video_session['frame_names']):
                logger.error(f"Frame index {frame_idx} out of range. Max: {len(video_session['frame_names'])-1}")
                return None, f"❌ Frame index {frame_idx} out of range"
            
            # Load frame image
            img_path = os.path.join(video_session['frames_dir'], video_session['frame_names'][frame_idx])
            logger.info(f"Loading frame: {img_path}")
            
            # Check if file exists
            if not os.path.exists(img_path):
                logger.error(f"Frame file does not exist: {img_path}")
                return None, f"❌ Frame file not found: {img_path}"

            # Load base image
            image = cv2.imread(img_path)
            if image is None:
                logger.error(f"Could not load image with cv2: {img_path}")
                return None, f"❌ Could not load image: {img_path}"
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get masks from segments (if they exist)
            masks_dict = {}
            if video_session["segments"] and frame_idx in video_session["segments"]:
                obj_ids = set(video_session["tracked_objects"].keys())
                masks_dict = {obj_id: mask for obj_id, mask in video_session["segments"][frame_idx].items() if obj_id in obj_ids}
            
            # Draw masks if available
            if masks_dict:
                logger.info(f"Found {len(masks_dict)} masks for frame {frame_idx}")
                
                for obj_id, mask in masks_dict.items():
                    logger.info(f"Processing mask for object {obj_id}, mask shape: {mask.shape if hasattr(mask, 'shape') else 'No shape'}")
                    
                    # Handle batch dimension properly
                    if len(mask.shape) == 3 and mask.shape[0] == 1:
                        mask = mask.squeeze(0)  # Remove batch dimension: (1, H, W) → (H, W)
                    elif mask.shape != (image_rgb.shape[0], image_rgb.shape[1]):
                        logger.warning(f"Mask shape {mask.shape} doesn't match image shape {image_rgb.shape[:2]}")
                        # Try to squeeze/reshape the mask
                        if mask.size == image_rgb.shape[0] * image_rgb.shape[1]:
                            mask = mask.reshape(image_rgb.shape[0], image_rgb.shape[1])
                            logger.info(f"Reshaped mask to {mask.shape}")
                        else:
                            logger.error(f"Cannot reshape mask - size mismatch. Mask size: {mask.size}, Expected: {image_rgb.shape[0] * image_rgb.shape[1]}")
                            continue  # Skip this mask

                    # Choose colors and opacity based on whether this is the selected object
                    if specific_obj_id is not None and obj_id == specific_obj_id:
                        # Highlighted appearance for selected object
                        mask_color = [0, 255, 0]  # Bright green
                        alpha = 0.4  # More opaque
                        border_color = (255, 255, 0)  # Yellow border
                        border_thickness = 3
                    else:
                        # Dimmed appearance for other objects
                        mask_color = [100, 100, 100]  # Gray
                        alpha = 0.2  # More transparent
                        border_color = (150, 150, 150)  # Light gray border
                        border_thickness = 1

                    # Create colored overlay
                    overlay = np.zeros_like(image_rgb)
                    overlay[mask] = mask_color

                    # Blend with original image
                    image_rgb = image_rgb.copy()
                    mask_pixels = image_rgb[mask]
                    overlay_pixels = overlay[mask]
                    
                    # Only blend if we have valid pixels selected by the mask
                    if mask_pixels.size > 0 and overlay_pixels.size > 0:
                        try:
                            # Additional shape check
                            if mask_pixels.shape != overlay_pixels.shape:
                                logger.error(f"Shape mismatch: mask_pixels {mask_pixels.shape} vs overlay_pixels {overlay_pixels.shape}")
                                continue
                            
                            blended_pixels = cv2.addWeighted(
                                mask_pixels, 1-alpha,
                                overlay_pixels, alpha,
                                0
                            )
                            image_rgb[mask] = blended_pixels
                            logger.info(f"Successfully blended mask for object {obj_id}")
                        except Exception as e:
                            logger.warning(f"Blending failed for object {obj_id}: {e}")
                            # Fallback: simple overlay without blending
                            if overlay_pixels.size > 0:
                                image_rgb[mask] = overlay_pixels
                    else:
                        logger.warning(f"Object {obj_id}: No pixels selected by mask (empty mask)")
                    
                    # Draw border around mask
                    mask_coords = np.where(mask)
                    if len(mask_coords[0]) > 0:
                        y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
                        x_min, x_max = mask_coords[1].min(), mask_coords[1].max()
                        cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), border_color, border_thickness)
                        
                        # Add object ID label
                        label_pos = (x_min, y_min - 10 if y_min > 20 else y_max + 25)
                        object_label = video_session['tracked_objects'][obj_id].get('label', f"Object_{obj_id}")
                        cv2.putText(image_rgb, object_label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, border_color, border_thickness)
            
            # Draw visual correction points (for ALL objects or just the selected one)
            if self.saved_points and frame_idx in self.saved_points:
                from PIL import ImageDraw
                result_pil = Image.fromarray(image_rgb)
                draw = ImageDraw.Draw(result_pil)
                
                for obj_id, points in self.saved_points[frame_idx].items():
                    # Only draw points for the specific object if one is selected
                    if specific_obj_id is not None and obj_id != specific_obj_id:
                        continue
                        
                    pos_points = points.get("positive", [])
                    neg_points = points.get("negative", [])
                    
                    # Choose colors based on whether this is the selected object
                    if specific_obj_id is not None and obj_id == specific_obj_id:
                        # Highlighted colors for selected object
                        pos_color = "red"
                        neg_color = "blue"
                        outline_color = "yellow"
                        radius = 10
                        outline_width = 3
                    else:
                        # Dimmed colors for other objects
                        pos_color = "#960000"  # Dark red
                        neg_color = "#000096"  # Dark blue
                        outline_color = "#C8C8C8"  # Gray
                        radius = 8
                        outline_width = 2
                    
                    # Draw positive points
                    for px, py in pos_points:
                        draw.ellipse([px-radius, py-radius, px+radius, py+radius], fill=pos_color)
                        draw.ellipse([px-radius-2, py-radius-2, px+radius+2, py+radius+2], outline=outline_color, width=outline_width)
                    
                    # Draw negative points
                    for nx, ny in neg_points:
                        draw.ellipse([nx-radius, ny-radius, nx+radius, ny+radius], fill=neg_color)
                        draw.ellipse([nx-radius-2, ny-radius-2, nx+radius+2, ny+radius+2], outline=outline_color, width=outline_width)
                
                image_rgb = np.array(result_pil)
            
            result_pil = Image.fromarray(image_rgb)
            
            # Create status message
            num_masks = len(masks_dict)
            num_objects = len(video_session.get("tracked_objects", {}))
            
            if specific_obj_id is not None:
                has_mask = specific_obj_id in masks_dict
                has_points = frame_idx in self.saved_points and specific_obj_id in self.saved_points.get(frame_idx, {})
                
                if has_mask and has_points:
                    status = f"Frame {frame_idx} - Object {specific_obj_id} highlighted (mask + points)"
                elif has_mask:
                    status = f"Frame {frame_idx} - Object {specific_obj_id} highlighted (mask only)"
                elif has_points:
                    status = f"Frame {frame_idx} - Object {specific_obj_id} highlighted (points only, no mask yet)"
                else:
                    status = f"Frame {frame_idx} - Object {specific_obj_id} (no data yet - add points to start tracking)"
            else:
                if num_masks > 0:
                    status = f"Frame {frame_idx} with {num_masks}/{num_objects} objects tracked"
                else:
                    status = f"Frame {frame_idx} - {num_objects} objects defined (no tracking masks yet)"
            
            return result_pil, status

        except Exception as e:
            logger.error(f"Error getting frame data for frame {frame_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, f"Error loading frame {frame_idx}: {e}"

    def process_visual_prompt_with_object_id(self, video_key, frame_idx, positive_points, negative_points, obj_id):
        """Process visual point annotations and generate SAM2 mask preview, with proper object management"""
        try:
            logger.info("=== USER ACTION: Process Visual Prompt (PREVIEW WITH SAM2 MASK) ===")
            logger.info(f"Frame index: {frame_idx}")
            logger.info(f"Positive points: {len(positive_points)}")
            logger.info(f"Negative points: {len(negative_points)}")
            logger.info(f"Requested Object ID: {obj_id}")
            
            if not positive_points and not negative_points:
                return "❌ Please add some points first", None, None
            
            # Check if video is loaded
            if video_key not in self.current_video_sessions.keys():
                return "❌ Please process a video first", None, None
            
            video_session = self.current_video_sessions[video_key]
            
            if frame_idx >= len(video_session['frame_names']):
                return f"❌ Frame index {frame_idx} out of range", None, None
            
            # Get current tracked objects and determine final object ID
            tracked_objects = video_session.get('tracked_objects', {})
            
            # Determine final object ID to use
            final_obj_id = None
            object_status = "new"  # "new", "existing"
            
            if obj_id is not None:
                # User specified an object ID
                if obj_id in tracked_objects:
                    # Object already exists - this is an update/correction
                    existing_obj = tracked_objects[obj_id]
                    object_status = "existing"
                    final_obj_id = obj_id
                    logger.info(f"Adding visual points to existing object {obj_id} ({existing_obj['label']})")
                else:
                    # User wants a specific new ID
                    final_obj_id = obj_id
                    object_status = "new"
                    logger.info(f"Creating new visual object with requested ID {obj_id}")
            else:
                # Auto-assign next available ID
                if tracked_objects:
                    final_obj_id = max(tracked_objects.keys()) + 1
                else:
                    final_obj_id = 1
                object_status = "auto_assigned"
                logger.info(f"Auto-assigned object ID {final_obj_id}")
            
            # Load the frame for preview
            img_path = os.path.join(video_session['frames_dir'], video_session['frame_names'][frame_idx])
            image = Image.open(img_path)
            
            # Generate SAM2 mask using a temporary inference state
            sam2_mask = None
            preview_image = image.copy()
            
            if video_session['inference_state']:
                try:
                    # Create a temporary lightweight inference state for preview only
                    logger.info(f"Creating temporary inference state for visual prompt on frame {frame_idx}")
                    temp_frame_dir = tempfile.mkdtemp()
                    current_frame_path = os.path.join(video_session['frames_dir'], video_session['frame_names'][frame_idx])
                    temp_frame_path = os.path.join(temp_frame_dir, f"{frame_idx:05d}.jpg")
                    shutil.copy2(current_frame_path, temp_frame_path)
                    
                    # Initialize temporary inference state with single frame
                    temp_inference_state = self.video_predictor.init_state(video_path=temp_frame_dir, async_loading_frames=True, offload_video_to_cpu=True, offload_state_to_cpu=True)
                    logger.info(f"Temporary inference state created for visual prompt on frame {frame_idx}")
                    
                    # Create temporary object for mask generation (negative ID to indicate preview)
                    temp_obj_id = -1
                    
                    # Combine points and labels
                    points = np.array(positive_points + negative_points, dtype=np.float32)
                    labels = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)
                    
                    logger.info(f"Generating SAM2 mask from {len(positive_points)} positive + {len(negative_points)} negative points")
                    
                    # Generate mask using SAM2 - ONLY for the current frame (no propagation)
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=temp_inference_state,
                        frame_idx=0,  # Use 0 since temp dir only has one frame
                        obj_id=temp_obj_id,
                        points=points,
                        labels=labels,
                    )
                    
                    # Extract the mask for THIS FRAME ONLY
                    if len(out_mask_logits) > 0:
                        mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                        if len(mask.shape) == 3:
                            mask = mask.squeeze(0)  # Remove batch dimension
                        sam2_mask = mask
                        
                        # Convert image to numpy for mask blending
                        image_np = np.array(preview_image)
                        
                        # Choose color based on object status
                        if object_status == "existing":
                            mask_color = [255, 165, 0]  # Orange for existing objects
                        else:
                            mask_color = [0, 255, 0]  # Green for new objects
                        
                        # Create mask overlay
                        mask_overlay = np.zeros_like(image_np)
                        mask_overlay[mask] = mask_color
                        
                        # Blend mask with image
                        mask_pil = Image.fromarray(mask_overlay.astype(np.uint8))
                        preview_image = Image.blend(preview_image, mask_pil, 0.3)
                        
                        logger.info("Generated SAM2 mask for visual prompt preview")
                    else:
                        logger.warning("No mask generated from visual points")
                        sam2_mask = np.zeros((image.height, image.width), dtype=bool)
                    
                    # Clean up temporary resources
                    shutil.rmtree(temp_frame_dir)
                    logger.info("Cleaned up temporary single-frame directory for visual prompt")
                    
                except Exception as e:
                    logger.error(f"Error in temporary single-frame processing for visual: {e}")
                    # Fallback: create empty mask
                    sam2_mask = np.zeros((image.height, image.width), dtype=bool)
                    logger.warning("Used fallback method for visual prompt - no SAM2 mask")
            
            # Draw annotation points on top of the mask
            draw = ImageDraw.Draw(preview_image)
            
            # Draw positive points (red) with white outline
            for px, py in positive_points:
                draw.ellipse([px-5, py-5, px+5, py+5], outline="red", fill="red", width=2)
                draw.ellipse([px-7, py-7, px+7, py+7], outline="white", width=2)
            
            # Draw negative points (blue) with white outline
            for nx, ny in negative_points:
                draw.ellipse([nx-5, ny-5, nx+5, ny+5], outline="blue", fill="blue", width=2)
                draw.ellipse([nx-7, ny-7, nx+7, ny+7], outline="white", width=2)
            
            # Add object ID label to preview
            if final_obj_id is not None:
                # Find a good position for the label
                label_x = int(np.mean([p[0] for p in positive_points + negative_points]))
                label_y = int(np.mean([p[1] for p in positive_points + negative_points])) - 30
                
                # Get object info for label
                if object_status == "existing":
                    existing_label = tracked_objects[final_obj_id]['label']
                    label_text = f"Obj {final_obj_id}: {existing_label} (UPDATE)"
                    label_color = "orange"
                else:
                    label_text = f"Obj {final_obj_id} (NEW)"
                    label_color = "green"
                
                # Draw label background
                text_bbox = draw.textbbox((label_x, label_y), label_text)
                draw.rectangle([text_bbox[0]-5, text_bbox[1]-2, text_bbox[2]+5, text_bbox[3]+2], 
                            fill="black", outline=label_color, width=2)
                draw.text((label_x, label_y), label_text, fill=label_color)
            
            # Actually add/update the object in tracked_objects here
            if object_status == "existing":
                # Update existing object with corrections
                existing_obj = tracked_objects[final_obj_id]
                if 'correction_points' not in existing_obj:
                    existing_obj['correction_points'] = {}
                if frame_idx not in existing_obj['correction_points']:
                    existing_obj['correction_points'][frame_idx] = {"positive": [], "negative": []}
                
                existing_obj['correction_points'][frame_idx]['positive'].extend(positive_points)
                existing_obj['correction_points'][frame_idx]['negative'].extend(negative_points)
                
                logger.info(f"Updated existing object {final_obj_id} with visual corrections")
                
            else:
                # Add new visual object using _add_tracked_object
                all_points = positive_points + negative_points
                point_labels = [1] * len(positive_points) + [0] * len(negative_points)
                
                self._add_tracked_object(
                    video_key=video_key,
                    obj_id=final_obj_id,
                    obj_type='visual_points',
                    frame_idx=frame_idx,
                    label=f"Visual_Object_{final_obj_id}",
                    points=all_points,
                    point_labels=point_labels,
                    mask=sam2_mask
                )
                
                logger.info(f"Added new visual object {final_obj_id} to tracking")
            
            # Also add to SAM2 tracking if inference state exists
            if video_session['inference_state']:
                try:
                    # Prepare points and labels for SAM2
                    points_array = np.array(positive_points + negative_points, dtype=np.float32)
                    labels_array = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)
                    
                    # Add to SAM2 tracking
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=video_session['inference_state'],
                        frame_idx=frame_idx,
                        obj_id=final_obj_id,
                        points=points_array,
                        labels=labels_array
                    )
                    logger.info(f"Added object {final_obj_id} to SAM2 tracking")
                except Exception as e:
                    logger.error(f"Error adding to SAM2 tracking: {e}")
            
            # Store detection data for potential future use
            detection_info = {
                "type": "visual_detection_with_mask",
                "frame_idx": frame_idx,
                "positive_points": positive_points,
                "negative_points": negative_points,
                "mask": sam2_mask,
                "obj_id": final_obj_id,
                "object_status": object_status
            }
            
            # Create status message based on object status
            if object_status == "existing":
                existing_obj = tracked_objects[final_obj_id]
                status = f"✅ UPDATED: Object {final_obj_id} ({existing_obj['label']}) with {len(positive_points)}+ {len(negative_points)}- points"
            else:
                status = f"✅ ADDED: Object {final_obj_id} with {len(positive_points)}+ {len(negative_points)}- points"
            
            return status, preview_image, detection_info
            
        except Exception as e:
            logger.error(f"Error processing visual prompt: {e}")
            return f"❌ Error: {str(e)}", None, None
    
    def segment_gaze_per_frame(self, video_key: str,  min_confidence: float = 0.0):
        """
        For every frame, use that frame's gaze points as SAM2 prompts and segment on that frame.
        Stores boolean masks per frame in video_session['segments'] (no propagation).
        """
        if video_key not in self.current_video_sessions:
            return "❌ Invalid video selection", None

        if not self.has_gaze_data():
            return "❌ No gaze data loaded", None

        vs = self.current_video_sessions[video_key]
        frames_dir = vs['frames_dir']
        frame_names = vs['frame_names']

        per_frame_segments = {}
        preview_samples = []

        for idx, fname in enumerate(frame_names):
            
            # Get video basename for directory-loaded gaze data
            video_basename = vs['video_name'] if hasattr(self.gaze_processor, 'video_file_mapping') else None
            gaze_points = self.gaze_processor.get_frame_gaze_points(
                idx, min_confidence=min_confidence, video_basename=video_basename
            )

            if not gaze_points:
                continue

            # Load frame + set image on SAM2 image predictor
            img_path = os.path.join(frames_dir, fname)
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)
            self.image_predictor.set_image(arr)

            # Segment each point
            masks_bool = []
            for (x, y) in gaze_points:
                m, s, l = self.image_predictor.predict(
                    point_coords=np.array([[x, y]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=False
                )
                if m is None or len(m) == 0:
                    continue
                # m shape: (1,H,W) or (H,W)
                mb = m[0] if m.ndim == 3 else m
                mb = (np.asarray(mb) > 0)
                masks_bool.append(mb)

            if not masks_bool:
                continue

            # Stack to (N,H,W)
            masks_bool = np.stack(masks_bool, axis=0)
            # Store for this frame
            per_frame_segments[idx] = {
                k+1: masks_bool[k] for k in range(masks_bool.shape[0])
            }

            for k in range(masks_bool.shape[0]):
                obj_id = k + 1
                self._add_tracked_object(video_key, obj_id, 'gaze', idx, f"Gaze_Object_{obj_id}")

            # Keep 1-2 preview frames
            if len(preview_samples) < 2:
                preview_img = self._create_segmentation_preview(
                    img, masks_bool, gaze_points, prompt_type="gaze"
                )
                preview_samples.append(preview_img)

        # Save into the session (no propagation)
        vs['segments'] = per_frame_segments

        total_frames = len(per_frame_segments)
        total_objs = sum(len(d) for d in per_frame_segments.values())
        status = f"✅ Gaze-per-frame segmentation complete: {total_objs} segments across {total_frames} frames"

        # Return a couple previews 
        return status, preview_samples



    def _search_new_obj_improved(self, masks_from_prev, mask_list, ratio=0.5, area_thresh=1000):
        """
        Improved version of search_new_obj from the reference script
        """
        new_mask_list = []
        
        if not masks_from_prev or len(masks_from_prev) == 0:
            return mask_list[:10]  # Return first 10 masks if no previous masks
        
        # Calculate mask_none - areas not covered by any previous mask
        first_mask = masks_from_prev[0]
        if len(first_mask.shape) == 3:
            mask_none = ~first_mask[0].copy()
        else:
            mask_none = ~first_mask.copy()
        
        for prev_mask in masks_from_prev[1:]:
            if len(prev_mask.shape) == 3:
                mask_none &= ~prev_mask[0]
            else:
                mask_none &= ~prev_mask
        
        # Find new objects that significantly overlap with uncovered areas
        for mask_dict in mask_list:
            seg = mask_dict['segmentation']
            if seg.sum() < area_thresh:  # Skip very small masks
                continue
                
            # Check if this mask covers significant uncovered area
            intersection_with_uncovered = (mask_none & seg).sum()
            if intersection_with_uncovered / seg.sum() > ratio:
                new_mask_list.append(mask_dict)
                # Update mask_none to exclude this new mask
                mask_none &= ~seg
        
        return new_mask_list

    def _calculate_coverage_ratio(self, frame_segments):
        """
        Calculate how much of the frame is covered by current segments
        """
        if not frame_segments:
            return 0.0
        
        # Get frame dimensions from first mask
        first_mask = next(iter(frame_segments.values()))
        if len(first_mask.shape) == 3:
            h, w = first_mask.shape[1], first_mask.shape[2]
            combined_mask = np.zeros((h, w), dtype=bool)
            for mask in frame_segments.values():
                combined_mask |= mask[0]
        else:
            h, w = first_mask.shape[0], first_mask.shape[1]
            combined_mask = np.zeros((h, w), dtype=bool)
            for mask in frame_segments.values():
                combined_mask |= mask
        
        coverage = combined_mask.sum() / (h * w)
        return coverage

    def _search_new_obj(self, masks_from_prev, mask_list, ratio=0.5, area_thresh=1000):
        """
        Find truly new objects by comparing against existing tracked masks
        """
        if not masks_from_prev:
            return mask_list
        
        new_mask_list = []
        
        # Create combined mask of all previously tracked objects
        combined_prev_mask = np.zeros_like(masks_from_prev[0], dtype=bool)
        for prev_mask in masks_from_prev:
            combined_prev_mask |= prev_mask
        
        # Find masks that cover significant new areas
        for mask_dict in mask_list:
            mask = mask_dict['segmentation']
            
            # Skip small masks
            if mask.sum() < area_thresh:
                continue
            
            # Calculate how much of this mask is in new areas
            new_area = mask & ~combined_prev_mask
            new_ratio = new_area.sum() / mask.sum() if mask.sum() > 0 else 0
            
            # Only add if it covers enough new area
            if new_ratio > ratio:
                new_mask_list.append(mask_dict)
                print(f"New object found: {new_ratio:.2f} new area ratio, {mask.sum()} total pixels")
        
        return new_mask_list
    def detect_objects_with_text(self, text_prompt, video_key, frame_idx, box_threshold=0.25, text_threshold=0.3):
        """
        Detect objects using GroundingDINO text prompts
        """
        try:
            if video_key not in self.current_video_sessions:
                return "❌ Invalid video selection", None, None
            
            video_session = self.current_video_sessions[video_key]
            
            if frame_idx >= len(video_session['frame_names']):
                return f"❌ Frame index {frame_idx} out of range", None, None
            
            # Load frame image
            img_path = os.path.join(video_session['frames_dir'], video_session['frame_names'][frame_idx])
            image = Image.open(img_path)
            
            # Format text prompt
            text = text_prompt.lower().strip()
            if not text.endswith('.'):
                text += '.'
            
            # Run GroundingDINO
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)
            
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]]
            )
            
            if len(results[0]["boxes"]) == 0:
                return f"❌ No objects found matching '{text_prompt}'", None, None
            
            boxes = results[0]["boxes"].cpu().numpy()
            labels = results[0]["labels"]
            scores = results[0]["scores"].cpu().numpy()
            
            # Create preview with bounding boxes
            preview_image = self._create_detection_preview(image, boxes, labels, scores)
            
            # Add gaze overlay if available
            if video_session.get('has_gaze_data', False):
                preview_image = self.gaze_processor.overlay_gaze_on_image(preview_image, frame_idx)
            
            detection_data = {
                "boxes": boxes.tolist(),
                "labels": labels,
                "scores": scores.tolist(),
                "frame_idx": frame_idx,
                "video_key": video_key,
                "prompt_type": "text"
            }
            
            video_name = self.current_video_sessions[video_key]['video_name']
            return f"✅ Detected {len(boxes)} objects in {video_name}", preview_image, detection_data
            
        except Exception as e:
            logger.error(f"Error in text detection: {e}")
            return f"❌ Error: {str(e)}", None, None

    def segment_detected_objects(self, video_key, frame_idx, detection_data):
        """
        Create masks for detected objects using SAM2
        """
        try:
            if video_key not in self.current_video_sessions:
                return "❌ Invalid video selection", None, None
            
            video_session = self.current_video_sessions[video_key]


            
            # Initialize inference state if not already done
            if video_session['inference_state'] is None:
                video_session['inference_state'] = self.video_predictor.init_state(
                    video_path=video_session['frames_dir'], async_loading_frames=True, offload_video_to_cpu=True, offload_state_to_cpu=True
                )

            boxes = detection_data.get("boxes", [])
            labels = detection_data.get("labels", [])
            
            if len(boxes) == 0:
                return "❌ No boxes found in segmentation data"
            
            # Add each detected object to track
            for object_id, (box, label) in enumerate(zip(boxes, labels), start=1):
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=video_session['inference_state'],
                    frame_idx=frame_idx,
                    obj_id=object_id,
                    box=box
                )
                self._add_tracked_object(video_key, object_id, 'box', frame_idx, label)
                
            # Load frame image
            img_path = os.path.join(video_session['frames_dir'], video_session['frame_names'][frame_idx])
            image = Image.open(img_path)
            
            # Debug: Print shapes to understand what we're getting
            print(f"Image shape: {np.array(image).shape}")
            print(f"Number of out_mask_logits: {len(out_mask_logits)}")
            for i, logits in enumerate(out_mask_logits):
                print(f"Logits {i} shape: {logits.shape}")

            masks = []
            for i, out_obj_id in enumerate(out_obj_ids):
                # Get the mask logits for this object
                mask_logits = out_mask_logits[i]
                
                # Convert to numpy and apply threshold
                mask_np = mask_logits.cpu().numpy()
                
                # Handle different possible shapes
                if mask_np.ndim == 3:
                    # Shape is likely (1, H, W) or (C, H, W)
                    if mask_np.shape[0] == 1:
                        mask_2d = mask_np[0]  # Remove first dimension
                    else:
                        mask_2d = mask_np[0]  # Take first channel
                elif mask_np.ndim == 2:
                    # Already 2D
                    mask_2d = mask_np
                else:
                    print(f"Unexpected mask shape: {mask_np.shape}")
                    continue
                
                # Apply threshold to get boolean mask
                boolean_mask = mask_2d > 0.0
                
                print(f"Final mask {i} shape: {boolean_mask.shape}")
                masks.append(boolean_mask)

            # Create preview with masks
            preview_image = self._create_segmentation_preview(
                image, masks, boxes, labels, prompt_type="text"
            )
                

            # Create preview with masks
            preview_image = self._create_segmentation_preview(
                image, masks, boxes, labels, prompt_type="text"
            )
            
            segmentation_data = {
                "masks": masks,
                "boxes": boxes,
                "labels": labels,
                "frame_idx": frame_idx,
                "video_key": video_key,
                "prompt_type": "text"
            }
            
            return f"✅ Created masks for {len(masks)} objects", preview_image, segmentation_data
            
        except Exception as e:
            logger.error(f"Error in object segmentation: {e}")
            return f"❌ Error: {str(e)}", None, None
    
    def reset_confidence_data(self, video_key):
        """Reset confidence data for a specific video session"""
        if video_key in self.current_video_sessions:
            video_session = self.current_video_sessions[video_key]
            video_session['confidence_data'] = {
                "frames": [],
                "iou_predictions": [],
                "occlusion_predictions": []
            }
            video_session['object_confidence_data'] = {}

    def apply_corrections_for_object(self, video_key, target_obj_id):
        """Apply saved corrections for a specific object only"""
        try:
            logger.info(f"=== USER ACTION: Apply Corrections for Object {target_obj_id} ===")
            
            video_session = self.current_video_sessions[video_key]

            # Check if we have any corrections for this object
            object_corrections = {}
            for frame_idx, objects in self.saved_points.items():
                if target_obj_id in objects:
                    points = objects[target_obj_id]
                    pos = points.get("positive", [])
                    neg = points.get("negative", [])
                    if pos or neg:
                        object_corrections[frame_idx] = {"positive": pos, "negative": neg}
            
            if not object_corrections:
                return f"❌ No saved corrections found for Object {target_obj_id}", None
        
            # Find earliest frame that needs correction
            start_anchor = min(list(object_corrections.keys())) if object_corrections else 0
            
            # Reset inference state
            video_session["inference_state"] = self.video_predictor.init_state(video_path=video_session['frames_dir'], async_loading_frames=True, offload_video_to_cpu=True, offload_state_to_cpu=True)
            self.reset_confidence_data(video_key)
            

            

            # Apply corrections for the target object
            corrections_applied = 0
            for frame_idx, corrections in object_corrections.items():
                pos_points = corrections["positive"]
                neg_points = corrections["negative"]
                
                # Combine positive and negative points
                all_points = []
                all_labels = []
                
                # Add positive points (label = 1)
                for px, py in pos_points:
                    all_points.append([px, py])
                    all_labels.append(1)
                
                # Add negative points (label = 0)
                for nx, ny in neg_points:
                    all_points.append([nx, ny])
                    all_labels.append(0)
                
                if all_points:
                    # Convert to numpy arrays
                    points_np = np.array(all_points, dtype=np.float32)
                    labels_np = np.array(all_labels, dtype=np.int32)
                    
                    # Add correction points to SAM2
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=video_session["inference_state"],
                        frame_idx=frame_idx,
                        obj_id=target_obj_id,
                        points=points_np,
                        labels=labels_np,
                    )
                    
                    corrections_applied += 1
                    logger.info(f"Applied corrections for Object {target_obj_id} at frame {frame_idx}: "
                            f"{len(pos_points)} positive, {len(neg_points)} negative points")
            
            # Propagate through the video to generate updated masks
            logger.info("Propagating corrections through video...")
            video_segments = {}
            
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                video_session["inference_state"]
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            # Update the stored video segments
            video_session["video_segments"] = video_segments
            

            # Backward pass
            for _ in self.video_predictor.propagate_in_video(
                video_session["inference_state"], start_frame_idx=start_anchor, reverse=True
            ):
                pass

            # Forward pass - collect per-object confidence data
            for out_fidx, out_ids, out_logits in self.video_predictor.propagate_in_video(
                 video_session["inference_state"], start_frame_idx=start_anchor
            ):
                video_session["segments"][out_fidx] = {
                    oid: (out_logits[i] > 0.0).cpu().numpy()
                    for i, oid in enumerate(out_ids)
                }

                # Update per-object confidence data
                for i, out_obj_id in enumerate(out_ids):
                    if out_obj_id not in video_session['object_confidence_data']:
                        video_session['object_confidence_data'][out_obj_id] = {
                            "frames": [],
                            "iou_predictions": [],
                            "occlusion_predictions": []
                        }
                    
                    if len(out_logits) > i:
                        mask_quality = torch.mean(torch.sigmoid(out_logits[i])).item()
                        obj_data = video_session['object_confidence_data'][out_obj_id]
                        if len(obj_data["iou_predictions"]) > 0:
                            prev_confidence = obj_data["iou_predictions"][-1]
                            occlusion_score = max(0.0, prev_confidence - mask_quality)
                        else:
                            occlusion_score = 0.0
                        
                        obj_data["frames"].append(out_fidx)
                        obj_data["iou_predictions"].append(mask_quality)
                        obj_data["occlusion_predictions"].append(occlusion_score)

                # Also update combined confidence data for backward compatibility
                if len(out_logits):
                    mq = torch.mean(torch.sigmoid(out_logits[0])).item()
                    prev =  video_session['confidence_data']["iou_predictions"][-1] if video_session['confidence_data']["iou_predictions"] else mq
                    occ = max(0.0, prev - mq)
                else:
                    mq, occ = 0.0, 1.0
                video_session['confidence_data']["frames"].append(out_fidx)
                video_session['confidence_data']["iou_predictions"].append(mq)
                video_session['confidence_data']["occlusion_predictions"].append(occ)

            logger.info(f"Applied {corrections_applied} correction points specifically for Object {target_obj_id}")
            return f"✅ Applied {corrections_applied} corrections for Object {target_obj_id}", None

        except Exception as e:
            logger.error(f"Error applying corrections for object {target_obj_id}: {e}")
            return f"❌ Error: {str(e)}", None
    
    # def track_video_with_text_segmentation(self, video_key, frame_idx, segmentation_data):
    def run_sam2_tracking(self, video_key, frame_idx, segmentation_data, prompt_type):
        """
        Execute tracking using text-based segmentation data
        """

        if prompt_type == 'text':
            try:
                if video_key not in self.current_video_sessions:
                    return "❌ Invalid video selection"
                
                video_session = self.current_video_sessions[video_key]
                
                boxes = segmentation_data["detection_data"][video_key].get("boxes", [])
                labels = segmentation_data["detection_data"][video_key].get("labels", [])
                
                if len(boxes) == 0:
                    return "❌ No boxes found in segmentation data"
                
                # Add each detected object to track
                for object_id, (box, label) in enumerate(zip(boxes, labels), start=1):
                    # _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    #     inference_state=video_session['inference_state'],
                    #     frame_idx=frame_idx,
                    #     obj_id=object_id,
                    #     box=box
                    # )
                    self._add_tracked_object(video_key, object_id, 'box', frame_idx, label)
               
            except Exception as e:
                logger.error(f"Error in text tracking: {e}")
                return f"❌ Tracking error: {str(e)}"
        elif prompt_type == 'visual':
            return self.handle_visual_tracking(video_key, frame_idx, segmentation_data)
        elif prompt_type == 'gaze':
            return "under implementation"
        else:
            return "Invalid prompt type"

        video_session = self.current_video_sessions[video_key]
        
        # Reset confidence data for this video session
        self.reset_confidence_data(video_key)

        # Propagate tracking
        logger.info("Starting SAM2 tracking propagation across all frames")
       
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(video_session["inference_state"], start_frame_idx=0):  # Start from frame 0
            video_session["segments"][out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > self.negative_threshold).cpu().numpy()  # Use configurable threshold
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
            # Extract confidence scores per object
            for i, out_obj_id in enumerate(out_obj_ids):
                # Initialize object confidence data if not exists
                if out_obj_id not in video_session['object_confidence_data']:
                    video_session['object_confidence_data'][out_obj_id] = {
                        "frames": [],
                        "iou_predictions": [],
                        "occlusion_predictions": []
                    }
                
                if len(out_mask_logits) > i:
                    # Get the confidence score
                    max_logit = torch.max(out_mask_logits[i]).item()
                    confidence_score = torch.sigmoid(torch.tensor(max_logit)).item()
                    
                    # Calculate IoU prediction
                    mask_quality = torch.mean(torch.sigmoid(out_mask_logits[i])).item()
                    
                    # Occlusion prediction
                    obj_data = video_session['object_confidence_data'][out_obj_id]
                    if len(obj_data["iou_predictions"]) > 0:
                        prev_confidence = obj_data["iou_predictions"][-1]
                        occlusion_score = max(0, prev_confidence - confidence_score)
                    else:
                        occlusion_score = 0.0
                    
                    obj_data["frames"].append(out_frame_idx)
                    obj_data["iou_predictions"].append(mask_quality)
                    obj_data["occlusion_predictions"].append(occlusion_score)
                else:
                    # No detection for this object in this frame
                    obj_data = video_session['object_confidence_data'][out_obj_id]
                    obj_data["frames"].append(out_frame_idx)
                    obj_data["iou_predictions"].append(0.0)
                    obj_data["occlusion_predictions"].append(1.0)
            
            # Also maintain combined confidence data for this video session
            if len(out_mask_logits) > 0:
                max_logit = torch.max(out_mask_logits[0]).item()
                confidence_score = torch.sigmoid(torch.tensor(max_logit)).item()
                mask_quality = torch.mean(torch.sigmoid(out_mask_logits[0])).item()
                
                if len(video_session['confidence_data']["iou_predictions"]) > 0:
                    prev_confidence = video_session['confidence_data']["iou_predictions"][-1]
                    occlusion_score = max(0, prev_confidence - confidence_score)
                else:
                    occlusion_score = 0.0
                
                video_session['confidence_data']["frames"].append(out_frame_idx)
                video_session['confidence_data']["iou_predictions"].append(mask_quality)
                video_session['confidence_data']["occlusion_predictions"].append(occlusion_score)
            else:
                video_session['confidence_data']["frames"].append(out_frame_idx)
                video_session['confidence_data']["iou_predictions"].append(0.0)
                video_session['confidence_data']["occlusion_predictions"].append(1.0)

        logger.info(f"SAM2 tracking completed! Processed {len(video_session['segments'])} frames")
        return f"✅ Tracking completed! Processed {len(video_session['segments'])} frames"
            

    def handle_visual_tracking(self, video_key, frame_idx, segmentation_data):
        """Handle visual point-based tracking"""
        try:

            if video_key not in self.current_video_sessions:
                return "❌ Invalid video selection"
            video_session = self.current_video_sessions[video_key]
            detection_info = segmentation_data.get("detection_data", {})
            frame_idx = detection_info.get("frame_idx", 0)
            positive_points = detection_info.get("positive_points", [])
            negative_points = detection_info.get("negative_points", [])
            sam2_mask = detection_info.get("mask")
            object_id = detection_info.get("obj_id", 1)
            
            if sam2_mask is None:
                return "❌ No mask found in visual detection data"
            
            # Convert points to numpy arrays for SAM2
            all_points = positive_points + negative_points
            all_labels = [1] * len(positive_points) + [0] * len(negative_points)
            
            if not all_points:
                # If no points, convert mask to points as fallback
                all_points, all_labels = self._mask_to_points(sam2_mask)
            
            # Add the visual prompt to SAM2 tracking
            out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=video_session['inference_state'],
                frame_idx=frame_idx,
                obj_id=object_id,
                points=np.array(all_points),
                labels=np.array(all_labels)
            )

            self._add_tracked_object(video_key, object_id, 'visual_points', frame_idx, f"Visual_Object_{object_id}")
            
            points_info = f"{len(positive_points)} positive, {len(negative_points)} negative points"
            return f"Added object {object_id} ({points_info}) to tracking"
            
        except Exception as e:
            logger.error(f"Error in single visual object tracking: {e}")
            return f"❌ Single visual object tracking error: {str(e)}"

    def _create_detection_preview(self, image, boxes, labels, scores):
        """
        Create preview image with detection boxes
        """
        try:
            from PIL import ImageDraw, ImageFont
            
            preview = image.copy()
            draw = ImageDraw.Draw(preview)
            
            # Try to use a better font
            try:
                font = ImageFont.truetype("arial.ttf", 26)
            except:
                font = ImageFont.load_default()
            
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Draw label with background
                text = f"{label} ({score:.2f})"
                bbox = draw.textbbox((x1, y1-25), text, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill="red")
                draw.text((x1, y1-25), text, fill="white", font=font)
            
            return preview
            
        except Exception as e:
            logger.error(f"Error creating detection preview: {e}")
            return image

    def _create_segmentation_preview(self, image, masks, points_or_boxes, labels=None, prompt_type="gaze"):
        """
        Create preview image with segmentation masks - Fixed version
        """
        try:
            import cv2
            from PIL import ImageDraw
            
            preview = image.copy()
            preview_array = np.array(preview)
            
            # Ensure we have the correct image dimensions
            img_height, img_width = preview_array.shape[:2]
            
            
            

            for i, mask in enumerate(masks):
                try:
                    # Ensure mask is 2D and matches image dimensions
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    
                    # Handle different mask shapes
                    if mask.ndim == 3:
                        if mask.shape[0] == 1:
                            mask = mask.squeeze(0)  # Remove batch dimension (1, H, W) -> (H, W)
                        elif mask.shape[-1] == 1:
                            mask = mask.squeeze(-1)  # Remove channel dimension (H, W, 1) -> (H, W)
                        else:
                            # Take first channel if multiple channels
                            mask = mask[:, :, 0] if mask.shape[-1] > 1 else mask[0, :, :]
                    
                    # Ensure mask is boolean
                    if mask.dtype != bool:
                        mask = mask > 0.5 if mask.dtype == np.float32 or mask.dtype == np.float64 else mask.astype(bool)
                    
                    # Verify mask dimensions match image
                    if mask.shape[:2] != (img_height, img_width):
                        logger.warning(f"Mask {i} shape {mask.shape} doesn't match image shape {(img_height, img_width)}")
                        # Resize mask to match image dimensions
                        mask = cv2.resize(mask.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST).astype(bool)
                    
                    # Apply mask overlay only if mask has content
                    if mask.any():
                        color = (255, 0, 0)
                        alpha = 0.2
                        
                        # Create mask overlay with proper dimensions
                        mask_overlay = np.zeros_like(preview_array)
                        
                        # Ensure we're working with 2D boolean mask
                        if mask.ndim == 2:
                            mask_pixels = mask
                        else:
                            mask_pixels = mask.reshape(img_height, img_width)
                        
                        # Apply color to mask pixels
                        mask_overlay[mask_pixels] = color
                        
                        # Blend with original image
                        # preview_array = cv2.addWeighted(preview_array, 1-alpha, mask_overlay, alpha, 0).astype(np.uint8)
                        preview_array = preview_array.copy()

                        preview_array[mask_pixels] = cv2.addWeighted(
                            preview_array[mask_pixels], 1-alpha,
                            mask_overlay[mask_pixels], alpha,
                            0
                        )
                        
                except Exception as mask_error:
                    logger.error(f"Error processing mask {i}: {mask_error}")
                    logger.error(f"Mask shape: {mask.shape if hasattr(mask, 'shape') else 'No shape'}")
                    logger.error(f"Mask type: {type(mask)}")
                    continue
            
            preview = Image.fromarray(preview_array)
            draw = ImageDraw.Draw(preview)
            
            # Draw prompts (points or boxes)
            if prompt_type == "gaze":
                # Draw gaze points
                for i, (x, y) in enumerate(points_or_boxes):
                    color = (0, 255, 0)
                    # Draw circle for gaze point
                    radius = 8
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                            outline=color, width=3, fill=None)
                    draw.text((x+10, y-10), f"Gaze {i+1}", fill=color)
            else:
                # Draw bounding boxes for text detection
                try:
                    for i, box in enumerate(points_or_boxes):
                        # Handle different box formats
                        if isinstance(box, (list, tuple)) and len(box) >= 4:
                            x1, y1, x2, y2 = box[:4]
                        elif hasattr(box, '__len__') and len(box) >= 4:
                            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        else:
                            logger.warning(f"Unexpected box format: {box}")
                            continue
                        
                        color = (255, 0, 0)
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        if labels and i < len(labels):
                            draw.text((x1, y1-20), f"{labels[i]}", fill=color)
                except Exception as box_error:
                    logger.error(f"Error drawing bounding boxes: {box_error}")
            
            return preview
            
        except Exception as e:
            logger.error(f"Error creating segmentation preview: {e}")
            logger.error(f"Image shape: {np.array(image).shape}")
            logger.error(f"Number of masks: {len(masks)}")
            if masks:
                logger.error(f"First mask shape: {masks[0].shape if hasattr(masks[0], 'shape') else 'No shape'}")
                logger.error(f"First mask type: {type(masks[0])}")
            return image
    
    def check_gaze_intersection_with_radius(self, gaze_points, mask, radius=5):
        """
        Check if gaze points intersect with mask using a circular radius around each point
        
        Args:
            gaze_points: List of (x, y) pixel coordinates
            mask: Binary mask (numpy array) - THE ACTUAL SEGMENTATION MASK
            radius: Radius in pixels around each gaze point
        
        Returns:
            bool: True if any gaze point (with radius) intersects the mask
        """
        if not gaze_points:
            return False
        
        # Ensure mask is 2-D boolean array
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim == 3:
            # common cases: (1,H,W) or (H,W,1)
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[-1] == 1:
                mask = mask[..., 0]
            else:
                # if it ever becomes RGB, collapse to any channel
                mask = mask.any(axis=-1)
        
        img_height, img_width = mask.shape[:2]
        
        for gaze_x, gaze_y in gaze_points:
            # Create circular area around gaze point
            y_coords, x_coords = np.ogrid[:img_height, :img_width]
            circle = (x_coords - gaze_x)**2 + (y_coords - gaze_y)**2 <= radius**2
            
            # Check if any mask pixels are within the circle
            # This checks if the ACTUAL SEGMENTATION MASK intersects with the gaze circle
            if np.any(mask & circle):
                return True
        
        return False
    
    def export_to_elan(self, video_key: str, output_path: str) -> bool:
        """
        Export tracking results to ELAN format for validation
        
        Args:
            video_key: Key of the video session to export
            output_path: Path where to save the .eaf file
        
        Returns:
            bool: True if export successful
        """
        try:
            if video_key not in self.current_video_sessions:
                logger.error(f"Video key {video_key} not found")
                return False
            
            video_session = self.current_video_sessions[video_key]
            video_segments = video_session.get('segments', {})
            tracked_objects = video_session.get('tracked_objects', {})
            
            if not video_segments:
                logger.warning(f"No tracking data found for {video_key}")
                return False
            
            if not tracked_objects:
                logger.warning(f"No tracked objects found for {video_key}")
                return False
            
            # Get video metadata
            video_name = video_session.get('video_name', video_key)
            video_path = video_session.get('video_path', '')
            
            # Calculate FPS and duration
            frame_count = len(video_session.get('frame_names', []))
            fps = 25  # Default, you might want to get this from actual video metadata
            duration_ms = int((frame_count / fps) * 1000)
            
            # Initialize ELAN exporter
            elan = ELANExporter(
                video_name=video_name,
                video_path=video_path,
                fps=fps,
                duration_ms=duration_ms,
                author="SAM2_Tracker"
            )
            
            # Debug: Print tracked objects
            logger.info(f"Tracked objects: {list(tracked_objects.keys())}")
            for obj_id, obj_info in tracked_objects.items():
                logger.info(f"  Object {obj_id}: label='{obj_info.get('label', 'MISSING')}'")
            
            # Prepare frame data for gaze intersections
            # Format: (frame_idx, object_id, object_label)
            gaze_frame_data = []
            
            logger.info(f"Processing {len(video_segments)} frames for gaze intersections")
            
            frames_with_gaze = 0
            frames_without_gaze = 0
            
            for frame_idx, objects_in_frame in video_segments.items():
                for obj_id, mask in objects_in_frame.items():
                    if obj_id not in tracked_objects:
                        logger.warning(f"Frame {frame_idx}: object {obj_id} not in tracked_objects, skipping")
                        continue
                    
                    obj_label = tracked_objects[obj_id].get('label', None)
                    
                    # Validate label
                    if obj_label is None or str(obj_label).strip() == "":
                        logger.warning(f"Frame {frame_idx}, Object {obj_id}: empty or None label, using fallback")
                        obj_label = f'object_{obj_id}'
                    
                    # Check if gaze intersects this object in this frame
                    if video_session.get('has_gaze_data', False):
                        # Get video basename for directory-loaded gaze
                        video_basename = video_session.get('video_name') if hasattr(self.gaze_processor, 'video_file_mapping') else None
                        gaze_points = self.gaze_processor.get_frame_gaze_points(frame_idx, video_basename=video_basename)
                        
                        if gaze_points:
                            # Check intersection with radius
                            gaze_radius = 5
                            intersects = self.check_gaze_intersection_with_radius(
                                gaze_points, mask, gaze_radius
                            )
                            
                            if intersects:
                                gaze_frame_data.append((frame_idx, obj_id, obj_label))
                                frames_with_gaze += 1
                            else:
                                frames_without_gaze += 1
                        else:
                            frames_without_gaze += 1
                    else:
                        logger.warning(f"No gaze data available for video {video_key}")
                        # If no gaze data, include all tracked objects
                        gaze_frame_data.append((frame_idx, obj_id, obj_label))
            
            logger.info(f"Gaze intersection stats: {frames_with_gaze} with gaze, {frames_without_gaze} without")
            logger.info(f"Total frame-object pairs with gaze: {len(gaze_frame_data)}")
            
            if not gaze_frame_data:
                logger.warning(f"No gaze intersections found for {video_key}")
                # Still save the file but it will be empty
            
            # Merge consecutive frames into segments
            gaze_segments = merge_consecutive_frames(gaze_frame_data)
            
            logger.info(f"Merged {len(gaze_frame_data)} gaze intersection frames into {len(gaze_segments)} segments")
            
            # Debug: Print first few segments
            for i, seg in enumerate(gaze_segments[:5]):
                logger.info(f"  Segment {i}: frames {seg['start_frame']}-{seg['end_frame']}, "
                           f"obj_id={seg['object_id']}, label='{seg['object_label']}'")
            
            # Add gaze intersection segments to ELAN
            segments_added = 0
            for gaze_seg in gaze_segments:
                label = gaze_seg.get('object_label', '').strip()
                if label:
                    elan.add_gaze_intersection(
                        start_frame=gaze_seg['start_frame'],
                        end_frame=gaze_seg['end_frame'],
                        object_label=label
                    )
                    segments_added += 1
                else:
                    logger.warning(f"Skipping segment with empty label: {gaze_seg}")
            
            logger.info(f"Added {segments_added} segments to ELAN exporter")
            
            # Save ELAN file
            success = elan.save(output_path)
            
            if success:
                logger.info(f"✅ ELAN export successful: {output_path}")
                logger.info(f"   Total gaze intersection segments: {len(gaze_segments)}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error exporting to ELAN: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


    def save_results_with_gaze(self, output_dir, save_video=True, save_csv=True, save_elan=True,
                               video_key=None, include_gaze_overlay=True, 
                               save_frames=True, save_confidence=True, gaze_radius=5,
                               save_masks=False):  # New parameter
        """Save results with gaze overlay, confidence data, gaze intersection analysis, and mask data for multiple videos"""
        try:
            logger.info("=== USER ACTION: Export Results ===")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Save video: {save_video}")
            logger.info(f"Save CSV: {save_csv}")
            logger.info(f"Save ELAN: {save_elan}")
            logger.info(f"Include gaze overlay: {include_gaze_overlay}")
            logger.info(f"Save frames: {save_frames}")
            logger.info(f"Save confidence: {save_confidence}")
            logger.info(f"Save masks: {save_masks}")
            
            if not self.current_video_sessions:
                return "❌ No tracking results to save"
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine which videos to save
            videos_to_save = [video_key] if video_key else list(self.current_video_sessions.keys())
            saved_results = []
            
            for vkey in videos_to_save:
                if vkey not in self.current_video_sessions:
                    continue
                
                video_session = self.current_video_sessions[vkey]
                video_segments = video_session['segments']  # This contains only gaze-intersecting objects
                
                video_name = self.current_video_sessions[vkey]['video_name']
                video_output_dir = os.path.join(output_dir, f"{video_name}")
                os.makedirs(video_output_dir, exist_ok=True)
                
                # Create masks directory if saving masks
                if save_masks:
                    masks_dir = os.path.join(video_output_dir, "masks")
                    os.makedirs(masks_dir, exist_ok=True)
                
                results_data = []
                confidence_rows = []
                
                # Get video dimensions for gaze coordinate conversion
                sample_frame_path = os.path.join(video_session['frames_dir'], video_session['frame_names'][0])
                sample_img = cv2.imread(sample_frame_path)
                if sample_img is None:
                    logger.error(f"Could not load sample frame: {sample_frame_path}")
                    continue
                img_height, img_width = sample_img.shape[:2]
                
                # Process ALL frames to maintain video length
                total_frames = len(video_session['frame_names'])
                for frame_idx in range(total_frames):
                    # Progress logging
                    if frame_idx == 0 or (frame_idx + 1) % 50 == 0 or frame_idx == total_frames - 1:
                        logger.info(f"Processing {video_name} frame {frame_idx+1}/{total_frames}")
                    
                    img_path = os.path.join(video_session['frames_dir'], video_session['frame_names'][frame_idx])
                    img = cv2.imread(img_path)
                    if img is None:
                        logger.warning(f"Could not load frame: {img_path}")
                        continue
                    
                    # Get tracking data for this frame (may be empty)
                    allowed_ids = set(video_session["tracked_objects"].keys())
                    segments = video_segments.get(frame_idx, {})
                    segments = {obj_id: mask for obj_id, mask in segments.items() if obj_id in allowed_ids}

                    object_ids = list(segments.keys())
                    masks = list(segments.values())

                    # Save masks as NPY files
                    if save_masks and masks:
                        frame_masks_dict = {}
                        for obj_id, mask in zip(object_ids, masks):
                            mask_2d = mask.squeeze() if mask.ndim == 3 else mask
                            frame_masks_dict[f"object_{obj_id}"] = mask_2d.astype(np.uint8)
                        
                        if frame_masks_dict:
                            mask_file_path = os.path.join(masks_dir, f"frame_{frame_idx:05d}_masks.npy")
                            np.savez_compressed(mask_file_path, **frame_masks_dict)
                    
                    # Get gaze data for this frame
                    gaze_points = []
                    gaze_intersections = []
                    gaze_info = {}
                    
                    if (video_session.get('has_gaze_data', False) and 
                        self.gaze_processor.gaze_data is not None):
                        
                        # Get video basename for directory-loaded gaze
                        video_basename = video_session.get('video_name') if hasattr(self.gaze_processor, 'video_file_mapping') else None
                        frame_gaze = self.gaze_processor.get_frame_gaze_points(frame_idx, video_basename=video_basename)
                        
                        if frame_gaze:
                            # Convert to pixel coordinates if normalized
                            gaze_points = []
                            for gx, gy in frame_gaze:
                                # Check if coordinates are normalized (0-1 range)
                                if 0 <= gx <= 1 and 0 <= gy <= 1:
                                    pixel_x = int(gx * img_width)
                                    pixel_y = int(gy * img_height)
                                else:
                                    pixel_x = int(gx)
                                    pixel_y = int(gy)
                                gaze_points.append((pixel_x, pixel_y))
                            
                            # Calculate gaze centroid
                            gaze_info = {
                                'gaze_points_count': len(frame_gaze),
                                'gaze_centroid_x': np.mean([p[0] for p in gaze_points]) if gaze_points else None,
                                'gaze_centroid_y': np.mean([p[1] for p in gaze_points]) if gaze_points else None
                            }
                    
                    # Process frame with tracking data (if any)
                    annotated_frame = img.copy()
                    
                    if len(masks) > 0:
                        # Normalize masks
                        masks = [m.squeeze() if getattr(m, "ndim", 0) == 3 else m for m in masks]
                        masks_array = np.stack(masks, axis=0)
                        
                        # Calculate bounding boxes from masks
                        bounding_boxes = []
                        for mask in masks:
                            if mask.ndim == 3:
                                mask = mask.squeeze()
                            
                            coords = np.where(mask)
                            if len(coords[0]) > 0:
                                y_min, y_max = coords[0].min(), coords[0].max()
                                x_min, x_max = coords[1].min(), coords[1].max()
                                bounding_boxes.append([x_min, y_min, x_max, y_max])
                            else:
                                bounding_boxes.append([0, 0, 1, 1])
                        
                        bounding_boxes = np.array(bounding_boxes, dtype=np.float32)
                        
                        # Check gaze-mask intersections (should be True since these are already filtered)
                        for obj_id, mask in zip(object_ids, masks):
                            intersection_found = True  # These objects are already gaze-filtered
                            if gaze_points:  # Double-check if needed
                                intersection_found = self.check_gaze_intersection_with_radius(
                                    gaze_points, mask, gaze_radius
                                )
                            gaze_intersections.append(intersection_found)
                        
                        # Create detections
                        detections = sv.Detections(
                            xyxy=bounding_boxes,
                            mask=masks_array,
                            class_id=np.array(object_ids, dtype=np.int32),
                        )
                        
                        # Save CSV data with gaze information
                        for i, (obj_id, bbox, intersects) in enumerate(zip(object_ids, detections.xyxy, gaze_intersections)):
                            x1, y1, x2, y2 = bbox
                            
                            row_data = {
                                'video': video_name,
                                'frame': frame_idx,
                                'object_id': obj_id,
                                'x1': float(x1), 'y1': float(y1),
                                'x2': float(x2), 'y2': float(y2),
                                'width': float(x2 - x1),
                                'height': float(y2 - y1),
                                'area': float((x2 - x1) * (y2 - y1)),
                                'gaze_intersection': intersects,
                                **gaze_info
                            }
                            results_data.append(row_data)
                        
                        # Collect confidence data if available
                        if save_confidence and video_session.get('object_confidence_data'):
                            for obj_id in object_ids:
                                if obj_id in video_session['object_confidence_data']:
                                    obj_data = video_session['object_confidence_data'][obj_id]
                                    frame_indices = obj_data.get("frames", [])
                                    if frame_idx in frame_indices:
                                        idx = frame_indices.index(frame_idx)
                                        iou_predictions = obj_data.get("iou_predictions", [])
                                        occlusion_predictions = obj_data.get("occlusion_predictions", [])
                                        
                                        if idx < len(iou_predictions) and idx < len(occlusion_predictions):
                                            confidence_rows.append({
                                                'video': video_name,
                                                'object_id': obj_id,
                                                'frame': frame_idx,
                                                'iou_prediction': float(iou_predictions[idx]),
                                                'occlusion_prediction': float(occlusion_predictions[idx]),
                                                'mask_quality': float(iou_predictions[idx])
                                        })
                        
                        if save_video:
                            # Annotate frame with tracking results and gaze intersection colors
                            # Draw masks with intersection-based colors
                            for i, (mask, obj_id) in enumerate(zip(masks, object_ids)):
                                # Check gaze intersection for this specific mask
                                intersects = False
                                if gaze_points:
                                    intersects = self.check_gaze_intersection_with_radius(
                                        gaze_points, mask, gaze_radius
                                    )
                                
                                # Set color based on gaze intersection
                                color = (0, 255, 0) if intersects else (0, 0, 255)  # Green if intersects, Blue (BGR format) for red
                                
                                mask_overlay = np.zeros_like(img)
                                if mask.ndim == 3:
                                    mask_2d = mask.squeeze()
                                else:
                                    mask_2d = mask
                                
                                # Ensure mask_2d is boolean
                                if mask_2d.dtype != bool:
                                    mask_2d = mask_2d > 0.5 if mask_2d.dtype in [np.float32, np.float64] else mask_2d.astype(bool)
                                    
                                mask_overlay[mask_2d] = color
                                alpha = 0.2
                                
                                annotated_frame = annotated_frame.copy()
                                
                                # Extract pixels selected by mask - FIX for cv2.addWeighted error
                                frame_pixels = annotated_frame[mask_2d]
                                overlay_pixels = mask_overlay[mask_2d]
                                
                                # Only blend if we have valid pixels selected by the mask
                                if frame_pixels.size > 0 and overlay_pixels.size > 0 and frame_pixels.shape == overlay_pixels.shape:
                                    try:
                                        blended_pixels = cv2.addWeighted(
                                            frame_pixels, 1-alpha,
                                            overlay_pixels, alpha,
                                            0
                                        )
                                        annotated_frame[mask_2d] = blended_pixels
                                    except Exception as e:
                                        logger.warning(f"Blending failed for object {obj_id}: {e}")
                                        # Fallback: simple overlay without blending
                                        if overlay_pixels.size > 0:
                                            annotated_frame[mask_2d] = overlay_pixels
                                elif overlay_pixels.size > 0:
                                    # If we can't blend but have overlay pixels, just use them
                                    annotated_frame[mask_2d] = overlay_pixels
                            
                            # Draw bounding boxes
                            box_annotator = sv.BoxAnnotator(thickness=2)
                            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                            
                            # Create labels
                            labels_for_display = []
                            for i, obj_id in enumerate(object_ids):
                                label = video_session['tracked_objects'][obj_id].get('label', f"Object_{obj_id}")
                                # Check gaze intersection again for label indicator
                                intersects = False
                                if gaze_points:
                                    intersects = self.check_gaze_intersection_with_radius(
                                        gaze_points, masks[i], gaze_radius
                                    )
                                gaze_indicator = " 👁️" if intersects else ""
                                labels_for_display.append(f"{label}")
                            
                            if labels_for_display:
                                label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2, text_padding=5)
                                annotated_frame = label_annotator.annotate(
                                    scene=annotated_frame, 
                                    detections=detections, 
                                    labels=labels_for_display
                                )
                    
                    # Add gaze overlay (regardless of whether there are tracking masks)
                    if (include_gaze_overlay and video_session.get('has_gaze_data', False) and 
                        self.gaze_processor.gaze_data is not None):
                        
                        # Convert to PIL for gaze overlay
                        pil_frame = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                        pil_frame = self.gaze_processor.overlay_gaze_on_image(
                            pil_frame, frame_idx, show_points=True
                        )
                        annotated_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                    else:
                        # Draw gaze points manually if no processor overlay
                        for gaze_x, gaze_y in gaze_points:
                            cv2.circle(annotated_frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)  # Red filled
                            cv2.circle(annotated_frame, (gaze_x, gaze_y), 7, (255, 255, 255), 2)  # White outline
                            cv2.circle(annotated_frame, (gaze_x, gaze_y), gaze_radius, (255, 255, 0), 1)  # Yellow radius
                    
                    # Save ALL frames (whether they have tracking data or not)
                    if save_video:
                        video_frames_dir = os.path.join(video_output_dir, "video_frames")
                        os.makedirs(video_frames_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(video_frames_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)
                        
                        # Save individual frames if requested
                        if save_frames:
                            cv2.imwrite(os.path.join(video_output_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)
                
                # Save CSV for this video
                if save_csv and results_data:
                    df = pd.DataFrame(results_data)
                    csv_path = os.path.join(video_output_dir, f"{video_name}_tracking_results.csv")
                    df.to_csv(csv_path, index=False)
                
                # Save confidence data for this video
                if save_confidence and confidence_rows:
                    confidence_df = pd.DataFrame(confidence_rows)
                    confidence_csv_path = os.path.join(video_output_dir, f"{video_name}_confidence_data.csv")
                    confidence_df.to_csv(confidence_csv_path, index=False)
                    logger.info(f"Saved confidence data for {video_name}")
                
                # Create video for this video
                if save_video:
                    video_frames_dir = os.path.join(video_output_dir, "video_frames")
                    if os.path.exists(video_frames_dir) and os.listdir(video_frames_dir):
                        output_video_path = os.path.join(video_output_dir, f"{video_name}_tracking_result.mp4")
                        create_video_from_images(video_frames_dir, output_video_path)
                        
                        # Clean up video frames if not saving individual frames
                        if not save_frames:
                            import shutil
                            shutil.rmtree(video_frames_dir)
                
                if save_elan:
                    elan_output_path = os.path.join(video_output_dir, f"{video_name}_tracking.eaf")
                    elan_success = self.export_to_elan(vkey, elan_output_path)
                    
                    if elan_success:
                        saved_results.append(f"✅ {video_name}: ELAN file saved")
                    else:
                        saved_results.append(f"❌ {video_name}: ELAN export failed")

                # Export gaze analysis if available
                if video_session.get('has_gaze_data', False):
                    gaze_files = self.gaze_processor.export_gaze_analysis(video_output_dir, video_name)
                    if gaze_files:
                        saved_results.append(f"✅ {video_name}: saved with gaze analysis to {video_output_dir}")
                    else:
                        saved_results.append(f"✅ {video_name}: saved to {video_output_dir}")
                else:
                    saved_results.append(f"✅ {video_name}: saved to {video_output_dir}")
            
            # Create combined CSV if multiple videos
            if len(videos_to_save) > 1 and save_csv:
                combined_data = []
                combined_confidence_data = []
                
                for vkey in videos_to_save:
                    video_name = self.current_video_sessions[vkey]['video_name']
                    
                    # Combine tracking results
                    csv_path = os.path.join(output_dir, video_name, f"{video_name}_tracking_results.csv")
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        combined_data.append(df)
                    
                    # Combine confidence data
                    if save_confidence:
                        confidence_path = os.path.join(output_dir, video_name, f"{video_name}_confidence_data.csv")
                        if os.path.exists(confidence_path):
                            conf_df = pd.read_csv(confidence_path)
                            combined_confidence_data.append(conf_df)
                
                # Save combined tracking results
                if combined_data:
                    combined_df = pd.concat(combined_data, ignore_index=True)
                    combined_csv_path = os.path.join(output_dir, "combined_tracking_results.csv")
                    combined_df.to_csv(combined_csv_path, index=False)
                    saved_results.append(f"✅ Combined tracking results: {combined_csv_path}")
                
                # Save combined confidence data
                if combined_confidence_data:
                    combined_conf_df = pd.concat(combined_confidence_data, ignore_index=True)
                    combined_conf_csv_path = os.path.join(output_dir, "combined_confidence_data.csv")
                    combined_conf_df.to_csv(combined_conf_csv_path, index=False)
                    saved_results.append(f"✅ Combined confidence data: {combined_conf_csv_path}")
            
            return "\n".join(saved_results)
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"❌ Export failed: {str(e)}"

    # Include all the existing methods that are still needed
    def get_video_metadata(self, video_path):
        """Extract detailed metadata from video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = frame_count / fps if fps > 0 else 0
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            
            cap.release()
            
            return {
                'width': width, 'height': height, 'fps': fps,
                'frame_count': frame_count, 'duration_sec': duration_sec,
                'file_size_mb': file_size_mb, 'resolution': f"{width}x{height}"
            }
            
        except Exception as e:
            logger.error(f"Error getting metadata for {video_path}: {e}")
            return None
    
    def extract_frames_from_video(self, video_path, output_dir):
        """Extract frames from a single video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            frame_idx = 0
            frame_names = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_filename = f"{frame_idx:05d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_names.append(frame_filename)
                frame_idx += 1
            
            cap.release()
            return frame_names
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            raise
    
    def get_video_files(self, path):
        """Get all video files from a path"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv', '.flv'}
        
        if os.path.isfile(path):
            if Path(path).suffix.lower() in video_extensions:
                return [path]
            else:
                raise ValueError(f"File {path} is not a supported video format")
        
        elif os.path.isdir(path):
            video_files = []
            for file in os.listdir(path):
                if Path(file).suffix.lower() in video_extensions:
                    video_files.append(os.path.join(path, file))
            
            if not video_files:
                raise ValueError(f"No video files found in directory: {path}")
            
            return sorted(video_files)
        
        else:
            raise ValueError(f"Path does not exist: {path}")
    
    def get_video_list(self):
        """Get list of available videos for selection"""
        return list(self.current_video_sessions.keys())
    
    def get_preview_frames_for_video(self, video_key, num_frames=5):
        """Get preview frames for a specific video with gaze overlay if available"""
        if video_key not in self.current_video_sessions:
            return []
        
        video_session = self.current_video_sessions[video_key]
        
        if os.path.isdir(video_session['video_path']):  # frames directory
            frame_names = video_session['frame_names'][:num_frames]
            preview_frames = []
            for i, frame_name in enumerate(frame_names):
                frame_path = os.path.join(video_session['frames_dir'], frame_name)
                try:
                    img = Image.open(frame_path)
                    
                    # Add gaze overlay if available
                    if video_session.get('has_gaze_data', False):
                        img = self.gaze_processor.overlay_gaze_on_image(img, i)
                    
                    img.thumbnail((200, 150), Image.Resampling.LANCZOS)
                    preview_frames.append(img)
                except Exception as e:
                    logger.error(f"Error loading frame {frame_path}: {e}")
            return preview_frames
        else:
            # Extract from video with gaze overlay
            return self.extract_preview_frames(video_session['video_path'], num_frames)
    
    def extract_preview_frames(self, video_path, num_frames=5):
        """Extract preview frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                cap.release()
                return []
            
            if frame_count <= num_frames:
                frame_indices = list(range(frame_count))
            else:
                frame_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
            
            preview_frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width = frame_rgb.shape[:2]
                    if width > 400:
                        scale = 400 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), 
                                             interpolation=cv2.INTER_LANCZOS4)
                    
                    img = Image.fromarray(frame_rgb)
                    
                    # Add gaze overlay if available
                    if self.gaze_processor.gaze_data is not None:
                        img = self.gaze_processor.overlay_gaze_on_image(img, frame_idx)
                    
                    preview_frames.append(img)
            
            cap.release()
            return preview_frames
            
        except Exception as e:
            logger.error(f"Error extracting preview frames: {e}")
            return []
    
    def switch_video_preview(self, video_key):
        """Switch preview to different video"""
        try:
            if not video_key or video_key not in self.current_video_sessions:
                return []
            
            preview_frames = self.get_preview_frames_for_video(video_key, 5)
            return preview_frames
            
        except Exception as e:
            logger.error(f"Error switching video preview: {e}")
            return []

    def save_correction_points(self, frame_idx, positive_points, negative_points, obj_id=None):
        """Save correction points for a specific frame and object"""
        try:
            logger.info(f"=== USER ACTION: Save Points button clicked ===")
            logger.info(f"Saving correction points for frame {frame_idx}")
            logger.info(f"Positive points: {positive_points}")
            logger.info(f"Negative points: {negative_points}")
            logger.info(f"Object ID: {obj_id}")
            
            if not positive_points and not negative_points:
                return "❌ No points to save", "No corrections saved"
            
            # Use current object ID if none specified
            if obj_id is None:
                obj_id = self.current_object_id
            
            # Initialize frame if not exists
            if frame_idx not in self.saved_points:
                self.saved_points[frame_idx] = {}
            
            # Initialize object if not exists
            if obj_id not in self.saved_points[frame_idx]:
                self.saved_points[frame_idx][obj_id] = {"positive": [], "negative": []}
            
            # Add new points to existing saved points
            self.saved_points[frame_idx][obj_id]["positive"].extend(positive_points)
            self.saved_points[frame_idx][obj_id]["negative"].extend(negative_points)
            
            # Create summary including initial prompts
            summary_lines = [f"💾 Saved {len(positive_points)} positive + {len(negative_points)} negative points for frame {frame_idx}, Object {obj_id}"]
            summary_lines.append("")
            
            # Show initial prompts
            initial_count = 0
            for p in self.initial_prompts:
                if p["type"] == "points":
                    initial_count += len(p["points"])
                elif p["type"] == "box":
                    initial_count += 1
            if initial_count > 0:
                summary_lines.append(f"🎯 Initial prompts: {initial_count} items")
            
            # Show saved points by frame and object
            total_saved = 0
            for f_idx, objects in self.saved_points.items():
                for o_id, points in objects.items():
                    pos_count = len(points["positive"])
                    neg_count = len(points["negative"])
                    if pos_count > 0 or neg_count > 0:
                        summary_lines.append(f"Frame {f_idx} Obj{o_id}: {pos_count}+ {neg_count}-")
                        total_saved += pos_count + neg_count
            
            summary_lines.insert(1, f"📊 Total saved: {total_saved} correction points")
            summary = "\n".join(summary_lines)
            
            # Create detailed points info for current frame
            current_pos = len(self.saved_points[frame_idx][obj_id]["positive"])
            current_neg = len(self.saved_points[frame_idx][obj_id]["negative"])
            points_info = f"Frame {frame_idx} Obj{obj_id}: {current_pos} positive, {current_neg} negative (SAVED)"
            
            return summary, points_info
            
        except Exception as e:
            logger.error(f"Error saving correction points: {e}")
            return f"❌ Error: {str(e)}", "Error occurred"
