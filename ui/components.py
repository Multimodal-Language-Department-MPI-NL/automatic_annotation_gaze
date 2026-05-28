import gradio as gr
import logging
from typing import Optional
from utils.video_utils import handle_image_click
import os
from PIL import Image

# Import gaze conversion functionality

logger = logging.getLogger(__name__)

# Initialize global storage
stored_segmentation_data = None

def create_ui(tracker, gaze_processor, gaze_converter):
    """UI with gaze integration"""
    
    # Custom CSS (same as original)
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    :root, .gradio-container, body {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                    Roboto, "Helvetica Neue", Arial, "Noto Sans", system-ui, sans-serif !important;
        line-height: 1.55;
    }
    body { font-size: 17px; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 10px; margin-bottom: 20px;
        color: white; text-align: center;
    }
    .main-header h1 { font-size: 2.5rem !important; }
    .main-header p { font-size: 1.2rem !important; }

    .gr-markdown h1 { font-size: 2.2rem !important; }
    .gr-markdown h2 { font-size: 1.9rem !important; }
    # .gr-markdown h3 { font-size: 1.6rem !important; }
    .gr-markdown h3 { 
    font-size: 2.0rem !important;  /* Updated - larger section headings */
    font-weight: 700 !important;
    margin-bottom: 15px !important;
    color: #374151 !important;
    }
    /* Make the main instruction text larger */
    .tab-content strong {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: #1f2937 !important;
        display: block !important;
        margin-bottom: 20px !important;
    }
    
    .tabs .tab-nav button, button[role="tab"] {
        font-size: 1.15rem !important; padding: 10px 16px !important;
        line-height: 1.2 !important;
    }
    .tabs .tab-nav button[aria-selected="true"],
    button[role="tab"][aria-selected="true"] {
        font-weight: 700 !important;
        box-shadow: inset 0 -2px 0 0 currentColor;
    }
    """

    logger = logging.getLogger(__name__)
    
    with gr.Blocks(
        title="Multi-Video Grounded SAM2 Object Tracker with Gaze Integration", 
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header
        with gr.Row():
            gr.HTML("""
                <div class="main-header">
                    <h1>🎯 Automatic Eye Tracking Data Annotation</h1>
                    
                </div>
            """)
        
        # State variables
        current_segmentation_data = gr.State(None)
        current_gaze_data = gr.State(None)
        detection_data = gr.State() 

        # State variables for visual annotation (same as mask correction)
        visual_positive_points = gr.State([])
        visual_negative_points = gr.State([]) 
        
        # Main workflow tabs
        with gr.Tabs() as main_tabs:
            
            # Step 1:Video Setup with Gaze
            with gr.Tab("📹 Step 1: Video & Gaze Setup") as setup_tab:
                # gr.HTML('<strong>Load videos and gaze data</strong>')
                gr.HTML('<div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 15px; color: #374151;">Load videos and gaze data</div>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 📁 Video Input")
                            
                            input_method = gr.Radio(
                                choices=["Single Video File", "Video Directory"],
                                value="Single Video File",
                                show_label=False
                                # label="Input Method"
                            )
                            
                            video_file = gr.File(
                                # label="📹 Upload Video", 
                                file_types=[".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"],
                                visible=True,
                                show_label=False
                            )
                            
                            video_dir = gr.Textbox(
                                # label="📂 Directory Path", 
                                placeholder="e.g., /path/to/videos/",
                                visible=False,
                                show_label=False
                            )
                        
                        # Gaze Data Section with Directory Support
                        with gr.Group():
                            gr.Markdown("### 👁️ Gaze Data")
                            gr.Markdown("*Upload gaze data for eye tracking integration*")
                            
                            gaze_csv_file = gr.File(
                                # label="📊 Gaze CSV File",
                                file_types=[".csv"],
                                visible=True,
                                show_label=False
                            )
                            
                            gaze_csv_dir = gr.Textbox(
                                # label="📂 Gaze CSV Directory Path",
                                placeholder="e.g., /path/to/gaze_csvs/",
                                info="Directory with CSV files matching video names (video.mp4 → video.csv)",
                                visible=False,
                                show_label=False
                            )
                            
                            with gr.Accordion("📋 CSV Format Requirements", open=False):
                                gr.Markdown("""
                                **Expected CSV columns:**
                                - `frame`: Frame number (0-indexed)
                                - `x`, `y`: Gaze coordinates in pixels
                               
                                
                                **Example:**
                                ```
                                frame,x,y
                                0,640,360
                                0,645,365
                                1,650,370
                                ```
                                """)
                            
                        # Processing controls
                        with gr.Group():
                            
                            process_btn = gr.Button(
                                "📊 Load Video + Gaze Data", 
                                variant="primary", 
                                size="lg"
                            )

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 📊 Processing Status")
                            video_status = gr.Textbox(
                                # label="Status", 
                                lines=3,
                                placeholder="Processing status will appear here...",
                                show_label=False
                            )
                        
            # Step 2: Object Detection/Segmentation with Gaze Prompts
            with gr.Tab("🎯 Step 2: Detection & Tracking") as detection_tab:
                gr.HTML('<strong>Define what to track</strong>')

                with gr.Row(visible=False) as video_selection_row:
                    video_selector = gr.Dropdown(
                        label="Choose Video for Processing",
                        choices=[], value=None
                    )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Prompt type selection
                        with gr.Group():
                            gr.Markdown("### 🔍 Prompt Configuration")
                            prompt_type = gr.Radio(
                                choices=[
                                    ("Textual", "text"),
                                    ("Visual", "visual"),
                                    ("Gaze Points", "gaze")
                                ],
                                value="text",
                                label="Prompt Type",
                                info="Choose your preferred input method"
                            )
                        
                        # Textual detection section
                        with gr.Group() as textual_group:
                            text_prompt = gr.Textbox(
                                label="Objects to Track",
                                placeholder="e.g., person, laptop, coffee cup",
                                info="Separate multiple objects with commas"
                            )
                            
                            with gr.Accordion("⚙️ Detection Settings", open=False):
                                with gr.Row():
                                    frame_idx = gr.Number(
                                        label="Reference Frame", value=0, 
                                        minimum=0, step=1,
                                        info="Frame to use for initial detection"

                                    )
                                    box_threshold = gr.Slider(
                                        0.1, 1.0, 0.25, 
                                        label="Detection Confidence",
                                        info="Higher values = fewer but more confident detections"
                                    )
                                    text_threshold = gr.Slider(
                                        0.1, 1.0, 0.3, 
                                        label="Text Matching",
                                         info="How well text must match detected objects"
                                    )                       

                        # Visual Annotation Section (reuses mask correction interface)
                        with gr.Group(visible=False) as visual_group:
                            gr.Markdown("**Visual Point Annotation**")
                            gr.Markdown("Click on the image to add positive/negative points")
                            
                            with gr.Row():
                                visual_frame_idx = gr.Number(
                                    label="Frame for Annotation", 
                                    value=0, 
                                    minimum=0, 
                                    step=1,
                                    info="Frame to annotate with points"
                                )
                                visual_correction_mode = gr.Radio(
                                    choices=["Add (+)", "Remove (-)"],
                                    value="Add (+)",
                                    label="Click Mode",
                                    info="Type of point to add when clicking"
                                )
                            
                            # ADD OBJECT MANAGEMENT FOR VISUAL PROMPTS
                            with gr.Row():
                                visual_current_object_id = gr.Number(
                                    label="Current Object ID",
                                    value=0,
                                    minimum=0,
                                    step=1,
                                    interactive=False
                                )
                            
                            
                            visual_object_info = gr.Textbox(
                                label="Object Info",
                                value="🎯 Current Object: 0 | 📊 Total Objects: 0",
                                interactive=False,
                                lines=2
                            )

                            
                            # Load frame button
                            load_visual_frame_btn = gr.Button("📷 Load Frame", variant="secondary")
                            
                            
                            with gr.Row():
                                clear_visual_points_btn = gr.Button("🧹 Clear Points", variant="secondary")
                                

                            visual_points_info = gr.Textbox(
                                label="Points Info",
                                value="No points added yet",
                                interactive=False
                            )
                            
                        #Gaze Group
                        with gr.Group(visible=False) as gaze_group:
                            gr.Markdown("### 👁️ Gaze-Based Detection")
                            
                            gaze_mode = gr.Radio(
                                choices=[
                                    ("Reference Frame Propagation", "reference"),
                                    ("Frame-by-Frame Detection", "frame_by_frame")
                                ],
                                value="reference",
                                label="Gaze Detection Mode",
                                info="Choose how to use gaze data for object detection"
                            )
                            
                            # Reference frame settings (visible when reference mode selected)
                            with gr.Group() as gaze_reference_group:
                                gaze_reference_frame = gr.Number(
                                    label="Reference Frame", 
                                    value=0, minimum=0, step=1,
                                    info="Frame to detect objects from gaze points, then propagate to all frames"
                                )
                                gr.Markdown("*Objects will be detected at gaze locations in this frame, then tracked across all frames*")
                            
                            # Frame-by-frame settings (visible when frame-by-frame mode selected)
                            with gr.Group(visible=False) as gaze_framewise_group:
                                gr.Markdown("*Objects will be detected at gaze locations independently for each frame*")
                                gaze_min_confidence = gr.Slider(
                                    0.0, 1.0, 0.3,
                                    label="Minimum Gaze Confidence",
                                    info="Minimum confidence threshold for gaze-based detection"
                                )
                        
                        with gr.Group():                         
                            # SINGLE detect button
                            detect_btn = gr.Button(
                                "🔍 Detect Objects", 
                                variant="primary",
                                size="lg"
                            )
                            
                                
                    
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 🎯 Processing Results")
                            
                            detection_status = gr.Textbox(
                                label="Processing Status", lines=3,
                                placeholder="Processing results will appear here..."
                            )
                            
                            with gr.Tabs():
                                with gr.Tab("🔍 Results Preview"):
                                    # REPLACE Gallery with Interactive Image for visual prompts
                                    with gr.Group(visible=True) as gallery_view:
                                        results_preview_gallery = gr.Gallery(
                                            label="Processing Results with Gaze Overlay",
                                            show_label=False,
                                            columns=[2, 3], rows=2, height=600,
                                            object_fit="contain", preview=True
                                        )
                                    
                                    # ADD Interactive Image for visual annotation
                                    with gr.Group(visible=False) as interactive_view:
                                        results_interactive_image = gr.Image(
                                            label="Click on image to add/remove points",
                                            height=600,
                                            interactive=True,
                                            elem_id="results_interactive_image"
                                        )
                                    
                                    with gr.Row():
                                        approve_btn = gr.Button("✅ Approve Detection and Start Tracking", variant="primary")
                                        refine_btn = gr.Button("🔧 Refine Masks", variant="secondary")
            
            # ✏️ Step 3: Mask Correction
            with gr.Tab("✏️ Step 3: Mask Correction", id="mask_correction"):

                def create_new_object_from_points(selected_video_key, frame_idx, pos_pts, neg_pts):
                    """Create a new object ID and add it to tracked objects"""
                    try:
                        if not pos_pts:
                            return "❌ Add at least one positive point first", update_object_choices(selected_video_key), "No positive points added", gr.update()
                        
                        vk = selected_video_key if selected_video_key else (tracker.get_video_list()[0] if tracker.get_video_list() else None)
                        if vk is None:
                            return "❌ Load a video first", update_object_choices(selected_video_key), "No video loaded", gr.update()
                        
                        # Generate new object ID
                        existing_objects = list(tracker.current_video_sessions[vk].get("tracked_objects", {}).keys())
                        new_obj_id = max(existing_objects, default=-1) + 1
                        
                        # Initialize tracked_objects if it doesn't exist
                        if "tracked_objects" not in tracker.current_video_sessions[vk]:
                            tracker.current_video_sessions[vk]["tracked_objects"] = {}
                        
                        # Add new object to tracked objects with basic structure
                        tracker.current_video_sessions[vk]["tracked_objects"][new_obj_id] = {
                            "created_from_points": True,
                            "creation_frame": int(frame_idx),
                            "initial_points": {"positive": pos_pts.copy(), "negative": neg_pts.copy()}
                        }
                        
                        # Save points for this frame and object
                        fidx = int(frame_idx)
                        tracker.saved_points.setdefault(fidx, {}).setdefault(new_obj_id, {"positive": [], "negative": []})
                        tracker.saved_points[fidx][new_obj_id]["positive"] = pos_pts.copy()
                        tracker.saved_points[fidx][new_obj_id]["negative"] = neg_pts.copy()
                        
                        status = f"✅ Created new Object {new_obj_id} from points on frame {fidx}"
                        points_info = f"Frame {fidx} Obj{new_obj_id}: {len(pos_pts)}+ {len(neg_pts)}- (NEW OBJECT)"
                        
                        # Update object choices and switch to new object
                        updated_choices = update_object_choices(selected_video_key)
                        
                        return status, updated_choices, points_info, gr.update(value=new_obj_id)
                        
                    except Exception as e:
                        logger.error(f"create_new_object_from_points error: {e}")
                        return f"❌ Error creating object: {e}", update_object_choices(selected_video_key), "Error", gr.update()
                
                def switch_object(selected_video_key, selected_obj_id):
                    """Switch to a different object for correction"""
                    try:
                        # Check if we have valid inputs
                        if not selected_video_key or selected_video_key not in tracker.current_video_sessions:
                            return "📹 No video selected - please load a video first", None, create_confidence_plot()
                        
                        # Handle case when no objects are available (dropdown is empty)
                        if selected_obj_id is None or selected_obj_id == "":
                            # Get basic frame without object highlighting
                            frame_image, status = tracker.get_tracked_frame_data(selected_video_key, 0, specific_obj_id=None)
                            return "📋 No objects available for correction - please detect and track objects first", frame_image, create_confidence_plot()
                        
                        video_session = tracker.current_video_sessions[selected_video_key]
                        obj_id = int(selected_obj_id)
                        
                        # Verify the object exists in tracked objects
                        if obj_id not in video_session.get("tracked_objects", {}):
                            frame_image, status = tracker.get_tracked_frame_data(selected_video_key, 0, specific_obj_id=None)
                            return f"❌ Object {obj_id} not found in tracked objects", frame_image, create_confidence_plot()
                        
                        # Get current frame for preview with specific object ID
                        current_frame = 0  # Default to first frame
                        
                        # Get frame data with specific object ID to show only that object's data
                        frame_image, status = tracker.get_tracked_frame_data(selected_video_key, current_frame, specific_obj_id=obj_id)
                        
                        # Create confidence plot for this specific object
                        confidence_plot = create_confidence_plot(selected_video_key, obj_id)
                        
                        return f"✅ Switched to Object {obj_id}", frame_image, confidence_plot
                        
                    except ValueError as e:
                        # Handle case where selected_obj_id can't be converted to int
                        frame_image, status = tracker.get_tracked_frame_data(selected_video_key, 0, specific_obj_id=None)
                        return f"❌ Invalid object ID format: {selected_obj_id}", frame_image, create_confidence_plot()
                    except Exception as e:
                        logger.error(f"Error switching object: {e}")
                        frame_image, status = tracker.get_tracked_frame_data(selected_video_key, 0, specific_obj_id=None)
                        return f"❌ Error: {str(e)}", frame_image, create_confidence_plot()

                def update_object_choices(selected_video_key):
                    """Update the object dropdown choices based on tracked objects"""
                    try:
                        # Check if selected_video_key exists and has tracked objects
                        if not selected_video_key or selected_video_key not in tracker.current_video_sessions:
                            return gr.update(visible=False, choices=[], value=None)
                        
                        # Update object info for this video
                        all_objects = list(tracker.current_video_sessions[selected_video_key]["tracked_objects"].keys())

                        if len(all_objects) > 0:
                            return gr.update(visible=True, choices=all_objects, value=all_objects[0])
                        else:
                            return gr.update(visible=False, choices=[], value=None)  # Remove the trailing comma
                    except Exception as e:
                        logger.error(f"Error updating object choices: {e}")
                        return gr.update(visible=False, choices=[], value=None)
                
                # UNUSED: # 8. Frame Timeline Slider Event Handler
                def update_frame_from_slider(frame_idx):
                    """Update frame when slider changes"""
                    # Get the currently selected object ID for highlighting
                    current_obj_id = tracker.current_object_id if tracker.current_object_id > 0 else None
                    frame_image, status = tracker.get_tracked_frame_data(int(frame_idx), specific_obj_id=current_obj_id)
                    # Clear points when using slider to change frames
                    points_info_msg = f"Frame {int(frame_idx)}: 0 positive, 0 negative (slider)"
                    return frame_image, status, [], [], points_info_msg

                # Clear visual points (reuse mask correction clear logic)
                def clear_visual_points(frame_idx):
                    """Clear all visual annotation points"""
                    return [], [], f"Frame {frame_idx}: No points added yet"

                # Load frame for visual annotation (reuse mask correction frame loading)
                def load_frame_for_visual_annotation(frame_idx):
                    """Load a specific frame for visual annotation in results panel"""
                    try:
                        video_keys = list(tracker.current_video_sessions.keys())

                        if len(video_keys) == 0:
                            return None, "❌ No video uploaded"
                        elif len(video_keys) > 1:
                            return None, "❌ Visual prompt feature works only with single video at a time. Please upload a single video instead of a video directory."
                        else:
                            video_session = tracker.current_video_sessions[video_keys[0]]
                            if frame_idx >= len(video_session["frame_names"]):
                                return None, f"❌ Frame index {frame_idx} out of range"
                            else:
                                img_path = os.path.join(video_session['frames_dir'], video_session['frame_names'][frame_idx])
                                image = Image.open(img_path)
                                return image, f"📷 Frame {frame_idx} loaded for annotation"
                    except Exception as e:
                        return None, f"❌ Error loading frame: {str(e)}" 

                
                def update_mask_correction_video_selector():
                    """Update video selector dropdown for mask correction when videos are loaded"""
                    try:
                        video_choices = tracker.get_video_list()
                        if not video_choices:
                            return (
                                gr.update(visible=False, choices=[])
                            )
                        
                        video_options = [(tracker.current_video_sessions[k]['video_name'], k) for k in video_choices]
                        show_selector = len(video_options) > 1
                        
                        if show_selector:
                            # Multiple videos - show dropdown
                            default_value = video_options[0][1]  # Select first video by default
                            current_video_name = tracker.current_video_sessions[default_value]['video_name']
                            return (
                                gr.update(visible=True, choices=video_options, value=default_value)
                                
                            )
                        else:
                            # Single video - hide dropdown but show current video
                            video_key = video_options[0][1]
                            video_name = video_options[0][0]
                            return (
                                gr.update(visible=False, choices=video_options, value=video_key)
                                
                            )
                    except Exception as e:
                        logger.error(f"Error updating mask correction video selector: {e}")
                        return (
                            gr.update(visible=False, choices=[])
                            
                        )

                def switch_mask_correction_video(selected_video_key):
                    """Switch to a different video for mask correction"""
                    try:
                        if not selected_video_key or selected_video_key not in tracker.current_video_sessions.keys():
                            return (
                                "❌ Invalid video selection",
                                None,
                                update_object_choices(selected_video_key),
                                gr.update(minimum=0, maximum=100, value=0),
                                create_confidence_plot(selected_video_key),
                                None
                            )
                        
                        # Update current video context for mask correction
                        video_name = tracker.current_video_sessions[selected_video_key]['video_name']
                        
                        # Get frame count for this video
                        video_session = tracker.current_video_sessions[selected_video_key]
                        max_frames = len(video_session['frame_names']) - 1

                        default_obj = int(list(video_session["tracked_objects"].keys())[0]) if video_session["tracked_objects"].keys() else None
                        
                        # Load first frame of selected video
                        frame_image, status = tracker.get_tracked_frame_data(
                            selected_video_key,
                            0,  
                            specific_obj_id=default_obj
                        )
                        
                        # Update timeline slider for this video
                        timeline_update = gr.update(minimum=0, maximum=max_frames, value=0)
                        
                        # Create confidence plot for current object in this video
                        confidence_plot = create_confidence_plot(
                            selected_video_key,
                            obj_id=default_obj
                        )
                        
                        return (
                            f"📹 Switched to: {video_name} | {status}",
                            frame_image,
                            update_object_choices(selected_video_key),
                            timeline_update,
                            confidence_plot
                        )

                    except Exception as e:
                        logger.error(f"Error switching mask correction video: {e}")
                        return (
                            f"❌ Error switching video: {str(e)}",
                            None,
                            update_object_choices(selected_video_key),
                            gr.update(minimum=0, maximum=100, value=0),
                            create_confidence_plot()
                        )

                def update_frame_from_slider_with_video(frame_idx, selected_video_key):
                    """Update frame when slider changes, accounting for selected video"""
                    # try:
                    # Use selected video key, fallback to single video or default
                    video_key = selected_video_key if selected_video_key else None
                    
                    # Get the currently selected object ID for highlighting
                    current_obj_id = tracker.current_object_id if tracker.current_object_id > 0 else None
                    
                    frame_image, status = tracker.get_tracked_frame_data(
                        video_key,
                        int(frame_idx), 
                        specific_obj_id=current_obj_id
                    )
                    
                    # Clear points when using slider to change frames
                    points_info_msg = f"Frame {int(frame_idx)}: 0 positive, 0 negative (slider)"
                    
                    return frame_image, status, [], [], points_info_msg
                        
                    # except Exception as e:
                    #     logger.error(f"Error updating frame from slider: {e}")
                    #     return None, f"❌ Error: {str(e)}", [], [], "Error loading frame"

                # def create_confidence_plot(obj_id=None):
                #     """Create confidence plot from tracking data for specific object or combined"""
                #     import matplotlib.pyplot as plt
                    
                #     # Determine which data to use
                #     if obj_id is not None and obj_id in tracker.object_confidence_data:
                #         confidence_data = tracker.object_confidence_data[obj_id]
                #         title_suffix = f" - Object {obj_id}"
                #     else:
                #         confidence_data = tracker.confidence_data
                #         title_suffix = " - All Objects Combined"
                    
                #     if not confidence_data or not confidence_data["frames"]:
                #         # Return empty plot if no data
                #         fig, ax = plt.subplots(figsize=(10, 4))
                #         ax.set_xlabel("Frame")
                #         ax.set_ylabel("Confidence Score")
                #         ax.set_title(f"SAM2 Confidence Score Over Time{title_suffix}")
                #         ax.text(0.5, 0.5, f"No tracking data available{title_suffix}", 
                #             ha='center', va='center', transform=ax.transAxes, fontsize=12)
                #         ax.set_xlim(0, 100)
                #         ax.set_ylim(0, 1)
                #         return fig
                    
                #     # Create plot with confidence data
                #     fig, ax = plt.subplots(figsize=(10, 4))
                    
                #     frames = confidence_data["frames"]
                #     iou_scores = confidence_data["iou_predictions"]
                #     occlusion_scores = confidence_data["occlusion_predictions"]
                    
                #     # Plot IoU predictions (mask quality)
                #     ax.plot(frames, iou_scores, 'b-', label='IoU Predictions (Mask Quality)', linewidth=2)
                    
                #     # Plot occlusion predictions
                #     ax.plot(frames, occlusion_scores, 'r--', label='Occlusion Predictions', linewidth=2)
                    
                #     ax.set_xlabel("Frame")
                #     ax.set_ylabel("Confidence Score")
                #     ax.set_title(f"SAM2 Confidence Score Over Time{title_suffix}")
                #     ax.legend()
                #     ax.grid(True, alpha=0.3)
                    
                #     # Set y-axis limits based on actual data range
                #     if iou_scores and occlusion_scores:
                #         min_val = min(min(iou_scores), min(occlusion_scores))
                #         max_val = max(max(iou_scores), max(occlusion_scores))
                #         # Add some padding
                #         padding = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
                #         ax.set_ylim(max(0, min_val - padding), min(1, max_val + padding))
                #     else:
                #         ax.set_ylim(0, 1)
                    
                #     plt.tight_layout()
                #     return fig

                def create_confidence_plot(video_key=None, obj_id=None):
                    """Create confidence plot using Plotly with per-video session data"""
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    # Handle video selection
                    if video_key is None:
                        # Use the first available video if none specified
                        available_videos = list(tracker.current_video_sessions.keys())
                        if not available_videos:
                            # Return empty plot if no videos
                            fig = go.Figure()
                            fig.add_annotation(
                                text="No videos loaded",
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                                showarrow=False, font=dict(size=16)
                            )
                            fig.update_layout(
                                title="SAM2 Confidence Score Over Time",
                                xaxis_title="Frame",
                                yaxis_title="Confidence Score",
                                xaxis=dict(range=[0, 100]),
                                yaxis=dict(range=[0, 1])
                            )
                            return fig
                        video_key = available_videos[0]
                    
                    # Validate video exists
                    if video_key not in tracker.current_video_sessions:
                        fig = go.Figure()
                        fig.add_annotation(
                            text=f"Video '{video_key}' not found",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, xanchor='center', yanchor='middle',
                            showarrow=False, font=dict(size=16)
                        )
                        fig.update_layout(
                            title="SAM2 Confidence Score Over Time",
                            xaxis_title="Frame",
                            yaxis_title="Confidence Score"
                        )
                        return fig
                    
                    video_session = tracker.current_video_sessions[video_key]
                    video_name = video_session.get('video_name', video_key)
                    
                    # Determine which confidence data to use
                    if obj_id is not None and obj_id in video_session.get('object_confidence_data', {}):
                        confidence_data = video_session['object_confidence_data'][obj_id]
                        title_suffix = f" - Object {obj_id}"
                        data_source = f"Object {obj_id}"
                    else:
                        confidence_data = video_session.get('confidence_data', {})
                        title_suffix = " - All Objects Combined"
                        data_source = "All Objects Combined"
                    
                    # Check if confidence data exists and has data
                    if not confidence_data or not confidence_data.get("frames"):
                        # Return empty plot if no data
                        fig = go.Figure()
                        fig.add_annotation(
                            text=f"No tracking data available for {video_name}<br>{data_source}",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, xanchor='center', yanchor='middle',
                            showarrow=False, font=dict(size=16)
                        )
                        fig.update_layout(
                            title=f"SAM2 Confidence Score Over Time - {video_name}{title_suffix}",
                            xaxis_title="Frame",
                            yaxis_title="Confidence Score",
                            xaxis=dict(range=[0, 100]),
                            yaxis=dict(range=[0, 1])
                        )
                        return fig
                    
                    # Extract data
                    frames = confidence_data["frames"]
                    iou_scores = confidence_data["iou_predictions"]
                    occlusion_scores = confidence_data["occlusion_predictions"]
                    
                    # Create plotly figure
                    fig = go.Figure()
                    
                    # Add IoU predictions trace
                    fig.add_trace(go.Scatter(
                        x=frames,
                        y=iou_scores,
                        mode='lines',
                        name='IoU Predictions (Mask Quality)',
                        line=dict(color='blue', width=2),
                        hovertemplate=f'Video: {video_name}<br>Frame: %{{x}}<br>IoU Score: %{{y:.3f}}<extra></extra>'
                    ))
                    
                    # Add occlusion predictions trace
                    fig.add_trace(go.Scatter(
                        x=frames,
                        y=occlusion_scores,
                        mode='lines',
                        name='Occlusion Predictions',
                        line=dict(color='red', width=2, dash='dash'),
                        hovertemplate=f'Video: {video_name}<br>Frame: %{{x}}<br>Occlusion Score: %{{y:.3f}}<extra></extra>'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"SAM2 Confidence Score Over Time - {video_name}{title_suffix}",
                        xaxis_title="Frame",
                        yaxis_title="Confidence Score",
                        legend=dict(x=0.02, y=0.98),
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    # Set y-axis range based on data
                    if iou_scores and occlusion_scores:
                        min_val = min(min(iou_scores), min(occlusion_scores))
                        max_val = max(max(iou_scores), max(occlusion_scores))
                        padding = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
                        fig.update_layout(yaxis=dict(range=[min_val, max_val]))
                    else:
                        fig.update_layout(yaxis=dict(range=[0,1]))
                    
                    return fig

                def delete_selected_object(selected_video_key, obj_id):
                    """Delete the selected object from tracking"""
                    try:
                        if not obj_id or obj_id is None:
                            return "❌ No object selected to delete", update_object_choices(selected_video_key)
                        
                        vk = selected_video_key if selected_video_key else (tracker.get_video_list()[0] if tracker.get_video_list() else None)
                        if vk is None:
                            return "❌ No video loaded", update_object_choices(selected_video_key)
                        
                        obj_id = int(obj_id)
                        
                        # Remove from tracked objects
                        if vk in tracker.current_video_sessions and "tracked_objects" in tracker.current_video_sessions[vk]:
                            if obj_id in tracker.current_video_sessions[vk]["tracked_objects"]:
                                del tracker.current_video_sessions[vk]["tracked_objects"][obj_id]
                        
                        # Remove from saved points
                        for frame_idx in tracker.saved_points:
                            if obj_id in tracker.saved_points[frame_idx]:
                                del tracker.saved_points[frame_idx][obj_id]
                        
                        # Remove from object confidence data if it exists
                        if hasattr(tracker, 'object_confidence_data') and obj_id in tracker.object_confidence_data:
                            del tracker.object_confidence_data[obj_id]
                        
                        status = f"✅ Deleted Object {obj_id}"
                        return status, update_object_choices(selected_video_key)
                        
                    except Exception as e:
                        logger.error(f"delete_selected_object error: {e}")
                        return f"❌ Error deleting object: {e}", update_object_choices(selected_video_key)


                # ---------- layout ----------
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📹 Video Selection")
                        mask_correction_video_selector = gr.Dropdown(
                            label="Select Video for Mask Correction",
                            choices=[],
                            value=None,
                            visible=False,
                            info="Choose video for corrections"
                        )

                        # object selection + info
                        obj_choice = gr.Dropdown(
                            label="Select Object",
                            choices=[],
                            value=None,
                            allow_custom_value=False,
                            info="Switch between tracked objects"
                        )


                        # frame slider
                        frame_timeline = gr.Slider(
                            minimum=0, maximum=900, value=0, step=1,
                            label="Frame Timeline",
                            info="Drag to navigate frames"
                        )

                        # click mode (Add/Remove/Delete)
                        correction_mode = gr.Radio(
                            choices=["Add (+)", "Remove (-)", "New Object"],  # Reorder
                            value="Add (+)",  # Change default from "New Object"
                            label="Click Mode",
                            info="Add positive/negative prompts, or create new object"
                        )

                        delete_obj_btn = gr.Button("🗑️ Delete Object", variant="secondary", scale=1)

                        with gr.Row():
                            clear_points_btn = gr.Button("🧹 Clear Points (frame)", variant="secondary")
                        #     save_points_btn = gr.Button("💾 Save Points", variant="secondary")

                         # live points & status
                        points_info = gr.Textbox(
                            label="Points Info",
                            value="Frame 0: 0 positive, 0 negative",
                            interactive=False
                        )

                        # with gr.Row():
                        #     create_obj_from_points_btn = gr.Button("➕ Create Object from Points", variant="secondary")

                        with gr.Row():
                            
                            apply_corrections_btn = gr.Button("✅ Apply Object Corrections (selected object)", variant="primary")

                            

                    with gr.Column(scale=2):

                        # interactive frame canvas
                        video_frame_image = gr.Image(
                            label="Video Frame — Click to Add/Remove/Delete Points",
                            height=520,
                            interactive=True,
                            elem_id="video_frame_for_clicks"
                        )

                        correction_status = gr.Textbox(label="Status", lines=3)

                        # confidence plot
                        confidence_plot = gr.Plot(label="SAM2 Confidence Over Time")

                        # in-memory click states
                        positive_points = gr.State([])
                        negative_points = gr.State([])

                # dropdown change → switch video
                mask_correction_video_selector.change(
                    fn=switch_mask_correction_video,
                    inputs=[mask_correction_video_selector],
                    outputs=[correction_status, video_frame_image, obj_choice, frame_timeline, confidence_plot]
                )


                def redraw_overlay(vk, frame_idx, pos_pts, neg_pts, selected_obj_id):
                    """Fetch a fresh base frame and draw current (+/−) points on top."""
                    # try:
                    vk = vk 
                    if not vk:
                        return None  # nothing to draw yet
                    obj = int(selected_obj_id) if selected_obj_id not in (None, "", "None") else None
                    base_img, _ = tracker.get_tracked_frame_data(vk, int(frame_idx), specific_obj_id=obj)
                    if base_img is None:
                        return None
                    from PIL import ImageDraw
                    im = base_img.copy()
                    d = ImageDraw.Draw(im)
                    # red = positive, blue = negative
                    for x, y in (pos_pts or []):
                        d.ellipse([x-5, y-5, x+5, y+5], outline="red", fill="red", width=2)
                    for x, y in (neg_pts or []):
                        d.ellipse([x-5, y-5, x+5, y+5], outline="blue", fill="blue", width=2)
                    return im
                    # except Exception as e:
                    #     logger.error(f"redraw_overlay error: {e}")
                    #     return None
                
                def clear_all_points(frame_idx):
                    """Clear all points for current frame"""
                    return [], [], f"Frame {frame_idx}: 0 positive, 0 negative (cleared)"

                # canvas click → add/remove/delete via your handler
                # video_frame_image.select(
                #     fn=handle_image_click,
                #     inputs=[video_frame_image, correction_mode, positive_points, negative_points, frame_timeline],
                #     outputs=[video_frame_image, positive_points, negative_points, points_info]
                # ).then(
                #     fn=lambda frame_idx, pos_pts, neg_pts, selected_obj: tracker.save_correction_points(int(frame_idx), pos_pts, neg_pts, int(selected_obj) if selected_obj is not None else None),
                #     inputs=[frame_timeline, positive_points, negative_points, obj_choice],
                #     outputs=[correction_status, points_info]
                # ).then(
                #     fn=redraw_overlay,
                #     inputs=[mask_correction_video_selector, frame_timeline, positive_points, negative_points, obj_choice],
                #     outputs=[video_frame_image]
                # )

                # Main click handler with automatic new object creation
                video_frame_image.select(
                    fn=handle_image_click,
                    inputs=[video_frame_image, correction_mode, positive_points, negative_points, frame_timeline],
                    outputs=[video_frame_image, positive_points, negative_points, points_info]
                ).then(
                    # Check if we're in "New Object" mode and have points - automatically create new object
                    fn=lambda mode, pos_pts, neg_pts, vk, frame_idx, obj_choice: (
                        create_new_object_from_points(vk, frame_idx, pos_pts, neg_pts) 
                        if mode == "New Object" and pos_pts 
                        else (gr.update(), gr.update(), gr.update(), gr.update())
                    ),
                    inputs=[correction_mode, positive_points, negative_points, mask_correction_video_selector, frame_timeline, obj_choice],
                    outputs=[correction_status, obj_choice, points_info, obj_choice]
                ).then(
                    # For non-"New Object" modes, save points to existing object
                    fn=lambda mode, frame_idx, pos_pts, neg_pts, selected_obj, vk: (
                        tracker.save_correction_points(int(frame_idx), pos_pts, neg_pts, int(selected_obj) if selected_obj is not None else None)
                        if mode != "New Object" and selected_obj is not None
                        else (gr.update(), gr.update())
                    ),
                    inputs=[correction_mode, frame_timeline, positive_points, negative_points, obj_choice, mask_correction_video_selector],
                    outputs=[correction_status, points_info]
                ).then(
                    fn=redraw_overlay,
                    inputs=[mask_correction_video_selector, frame_timeline, positive_points, negative_points, obj_choice],
                    outputs=[video_frame_image]
                )

                # # clear all in-memory points
                clear_points_btn.click(
                    fn=clear_all_points,
                    inputs=[frame_timeline],
                    outputs=[positive_points, negative_points, points_info]
                ).then(
                    fn=redraw_overlay,
                    inputs=[mask_correction_video_selector, frame_timeline, positive_points, negative_points, obj_choice],
                    outputs=[video_frame_image]
                )

                # Delete object button handler
                delete_obj_btn.click(
                    fn=delete_selected_object,
                    inputs=[mask_correction_video_selector, obj_choice],
                    outputs=[correction_status, obj_choice]
                ).then(
                    # Refresh the frame view after deletion
                    fn=lambda vk, frame_idx: tracker.get_tracked_frame_data(
                        vk if vk else (tracker.get_video_list()[0] if tracker.get_video_list() else None),
                        int(frame_idx), 
                        specific_obj_id=None  # Show all objects after deletion
                    )[0],
                    inputs=[mask_correction_video_selector, frame_timeline],
                    outputs=[video_frame_image]
                )

                # apply_corrections_btn.click(
                #     fn=lambda vk, oid: tracker.apply_corrections_for_object(
                #         vk if vk else (tracker.get_video_list()[0] if tracker.get_video_list() else None), 
                #         int(oid)
                #     ),
                #     inputs=[mask_correction_video_selector, obj_choice],
                #     outputs=[correction_status, gr.State()]  # second output placeholder (ignored)
                # ).then(
                #     # REFRESH THE FRAME IMAGE after corrections are applied
                #     fn=lambda vk, frame_idx, oid: tracker.get_tracked_frame_data(
                #         vk if vk else (tracker.get_video_list()[0] if tracker.get_video_list() else None),
                #         int(frame_idx), 
                #         specific_obj_id=int(oid) if oid else None
                #     )[0],  # Return only the image
                #     inputs=[mask_correction_video_selector, frame_timeline, obj_choice],
                #     outputs=[video_frame_image]
                # ).then(
                #     fn=lambda vk, oid=None: create_confidence_plot(
                #         vk, obj_id=int(oid) if oid else None
                #     ),
                #     inputs=[mask_correction_video_selector, obj_choice],
                #     outputs=[confidence_plot]
                # )

                apply_corrections_btn.click(
                    fn=lambda vk, oid: tracker.apply_corrections_for_object(
                        vk if vk else (tracker.get_video_list()[0] if tracker.get_video_list() else None), 
                        int(oid)
                    ),
                    inputs=[mask_correction_video_selector, obj_choice],
                    outputs=[correction_status, gr.State()]  # second output placeholder (ignored)
                ).then(
                    # REFRESH THE FRAME IMAGE after corrections are applied - show ALL objects
                    fn=lambda vk, frame_idx: tracker.get_tracked_frame_data(
                        vk if vk else (tracker.get_video_list()[0] if tracker.get_video_list() else None),
                        int(frame_idx), 
                        specific_obj_id=None  # Changed from int(oid) to None to show all objects
                    )[0],  # Return only the image
                    inputs=[mask_correction_video_selector, frame_timeline],  # Removed obj_choice from inputs
                    outputs=[video_frame_image]
                ).then(
                    fn=lambda vk, oid=None: create_confidence_plot(
                        vk, obj_id=int(oid) if oid else None
                    ),
                    inputs=[mask_correction_video_selector, obj_choice],
                    outputs=[confidence_plot]
                )

            demo.load(
                fn=update_mask_correction_video_selector,
                inputs=[],
                outputs=[mask_correction_video_selector]
            )

            video_choices = tracker.get_video_list()

            if video_choices:
                demo.load(
                    fn=update_object_choices,
                    inputs=tracker.get_video_list()[0],
                    outputs=[obj_choice]
                )
           
       
            # Step 4: Results & Export with Gaze
            with gr.Tab("💾 Step 4: Results & Export") as export_tab:
                gr.HTML('<strong>Export results with gaze analysis</strong>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 📁 Export Configuration")
                            
                            output_dir = gr.Textbox(
                                label="📂 Output Directory", 
                                value="./tracking_results"
                            )
                            
                            export_formats = gr.CheckboxGroup(
                                choices=[
                                    ("📹 Annotated Video (with gaze)", "video"),
                                    ("📊 Tracking Data (CSV)", "csv"),
                                    ("📝 ELAN Annotation File", "elan")
                                ],
                                value=["video", "elan"],
                                label="Export Formats"
                            )
                        
                        # gaze export options
                        with gr.Group():
                            gr.Markdown("### 👁️ Gaze Export Options")
                            
                            gaze_exports = gr.CheckboxGroup(
                                choices=[
                                    ("📊 Gaze-object analysis", "gaze_object")
                                ],
                                value=[ "gaze_object"],
                                label="Gaze Analysis Exports"
                            )
                            
                            include_gaze_overlay = gr.Checkbox(
                                label="Include Gaze Overlay on Videos",
                                value=True,
                                info="Show gaze points on output videos"
                            )
                        
                        export_btn = gr.Button(
                            "💾 Export Results", 
                            variant="primary", size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 📊 Export Status")
                            
                            export_status = gr.Textbox(lines=4,
                                placeholder="Export status will appear here..."
                            )
                            
                           
        
        # Event Handlers
        
        def toggle_input_visibility(method):
            """Toggle visibility of input fields based on method selection"""
            is_directory = (method == "Video Directory")
            return (
                gr.update(visible=not is_directory),  # video_file
                gr.update(visible=is_directory),      # video_dir
                gr.update(visible=not is_directory),  # gaze_csv_file
                gr.update(visible=is_directory)       # gaze_csv_dir
            )

        def toggle_prompt_visibility(prompt_type):
            """Toggle visibility based on prompt type and switch right panel view"""
            is_visual = (prompt_type == "visual")

            return (
                gr.update(visible=prompt_type=="text"),      # textual_group
                gr.update(visible=prompt_type=="visual"),    # visual_group  
                gr.update(visible=prompt_type=="gaze"),      # gaze_group
                gr.update(visible=not is_visual),            # gallery_view (hide for visual)
                gr.update(visible=is_visual)                 # interactive_view (show for visual)
            )

        # Visibility toggle for gaze mode
        def toggle_gaze_mode_visibility(gaze_mode):
            """Toggle visibility based on gaze mode"""
            is_reference = (gaze_mode == "reference")
            return (
                gr.update(visible=is_reference),     # gaze_reference_group
                gr.update(visible=not is_reference)  # gaze_framewise_group
            )

        def handle_video_and_gaze_processing(video_file, video_dir, input_method, 
                                   gaze_csv_file, gaze_csv_dir, fps=None):
            """Handles both video and gaze data with automatic format detection and conversion"""
            converted_files = []  # Track temporary files for cleanup
            try:
                # Determine gaze input based on method
                gaze_input = None
                if input_method == "Single Video File":
                    gaze_input = gaze_csv_file
                else:  # Video Directory
                    gaze_input = gaze_csv_dir
                
                # Process gaze data if provided
                processed_gaze_input = None
                gaze_loaded = False
                if gaze_input == '':
                    gaze_input = None
                if gaze_input is not None:
                    if input_method == "Single Video File":
                        # Handle different Gradio file formats 
                        if isinstance(gaze_input, list) and len(gaze_input) > 0:
                            input_path = gaze_input[0] if isinstance(gaze_input[0], str) else gaze_input[0].name
                        elif hasattr(gaze_input, 'name'):
                            input_path = gaze_input.name
                        elif isinstance(gaze_input, str):
                            input_path = gaze_input
                        else:
                            return "❌ Invalid gaze file format received", gr.update(visible=False), gr.update(choices=[])
                        
                        # Auto-detect file type and convert if needed
                        input_lower = input_path.lower()
                        if input_lower.endswith('.json'):
                            try:
                                # Convert using video framerate if available
                                converted_path = gaze_converter.convert_gaze_file(input_path, fps)
                                converted_files.append(converted_path)
                                processed_gaze_input = converted_path
                            except Exception as e:
                                return f"❌ Error converting JSON file: {str(e)}", gr.update(visible=False), gr.update(choices=[])
                        
                        elif input_lower.endswith('.csv'):
                            try:
                                # Check if it's already in the correct format or needs conversion
                                import pandas as pd
                                test_df = pd.read_csv(input_path, nrows=5)  # Read just a few rows to check format
                                expected_cols = ['frame', 'x', 'y']
                                
                                if all(col in test_df.columns for col in expected_cols):
                                    # Already in correct format
                                    processed_gaze_input = input_path
                                else:
                                    # Needs conversion from Tobii format
                                    converted_path = gaze_converter.convert_gaze_file(input_path, fps)
                                    converted_files.append(converted_path)
                                    processed_gaze_input = converted_path
                                    
                            except Exception as e:
                                return f"❌ Error processing CSV file: {str(e)}", gr.update(visible=False), gr.update(choices=[])
                        else:
                            return "❌ Unsupported file format. Please upload a JSON (Pupil Neon) or CSV (Tobii/standardized) file", gr.update(visible=False), gr.update(choices=[])
                    
                    else:
                        # Directory mode - process all gaze files in directory
                        if os.path.isdir(gaze_input):
                            processed_gaze_input = tempfile.mkdtemp(prefix='gaze_converted_dir_')
                            
                            gaze_files = []
                            for ext in ['.csv', '.json']:
                                gaze_files.extend([f for f in os.listdir(gaze_input) if f.lower().endswith(ext)])
                            
                            if not gaze_files:
                                return f"❌ No gaze files found in directory: {gaze_input}", gr.update(visible=False), gr.update(choices=[])
                            
                            conversion_errors = []
                            for gaze_file in gaze_files:
                                input_path = os.path.join(gaze_input, gaze_file)
                                input_lower = input_path.lower()
                                
                                try:
                                    if input_lower.endswith('.json'):
                                        # Convert JSON file
                                        converted_path = gaze_converter.convert_gaze_file(input_path, fps)
                                        # Move to processed directory with .csv extension
                                        base_name = os.path.splitext(gaze_file)[0] + '.csv'
                                        final_path = os.path.join(processed_gaze_input, base_name)
                                        import shutil
                                        shutil.move(converted_path, final_path)
                                        
                                    elif input_lower.endswith('.csv'):
                                        # Check if CSV needs conversion
                                        import pandas as pd
                                        test_df = pd.read_csv(input_path, nrows=5)
                                        expected_cols = ['frame', 'x', 'y']
                                        
                                        if all(col in test_df.columns for col in expected_cols):
                                            # Already in correct format - copy directly
                                            output_path = os.path.join(processed_gaze_input, gaze_file)
                                            import shutil
                                            shutil.copy2(input_path, output_path)
                                        else:
                                            # Needs conversion from Tobii format
                                            converted_path = gaze_converter.convert_gaze_file(input_path, fps)
                                            final_path = os.path.join(processed_gaze_input, gaze_file)
                                            import shutil
                                            shutil.move(converted_path, final_path)
                                            
                                except Exception as e:
                                    conversion_errors.append(f"Failed to process {gaze_file}: {str(e)}")
                                    continue
                            
                            if conversion_errors:
                                error_msg = "❌ Some gaze files could not be processed:\n" + "\n".join(conversion_errors)
                                return error_msg, gr.update(visible=False), gr.update(choices=[])
                        else:
                            return f"❌ Gaze directory not found: {gaze_input}", gr.update(visible=False), gr.update(choices=[])
                    
                    # Load processed gaze data
                    if input_method == "Single Video File":
                        success, message = gaze_processor.load_gaze_csv(processed_gaze_input)
                        
                    else:
                        success, message = gaze_processor.load_gaze_directory(processed_gaze_input)
                        
                    
                    if not success:
                        return f"❌ {message}", gr.update(visible=False), gr.update(choices=[])
                    else:
                        gaze_loaded=True

                
                
                # Process video with gaze integration
                if input_method == "Single Video File":
                    status = tracker.process_video_input(video_file, None, gaze_loaded)
                else:
                    status = tracker.process_video_input(None, video_dir, gaze_loaded)
                
                # Update UI components
                video_choices = tracker.get_video_list()
                video_options = [(tracker.current_video_sessions[k]['video_name'], k) for k in video_choices]
                show_selection = len(video_options) > 1
                
                if show_selection:
                    video_options = [("All Videos (batch)", "__ALL__")] + video_options
                    default_value = "__ALL__"
                else:
                    default_value = video_options[0][1] if video_options else None
                
                return (
                    status,
                    gr.update(visible=show_selection),  # For video_selection_row
                    gr.update(choices=video_options, value=default_value)  # For video_selector
                )
                
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                
                logger.error(f"Error processing video/gaze: {error_trace}")
                return (
                    f"❌ Error: {str(e)}",
                    gr.update(visible=False), 
                    gr.update(choices=[])
                )
            
            finally:
                # Cleanup temporary converted files
                for temp_file in converted_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        logger.error(f"Could not cleanup temporary file {temp_file}: {e}")



        def process_objects(text_prompt, frame_idx, box_thresh, text_thresh,
                   video_key, prompt_type, 
                   visual_frame_idx, visual_pos_points, visual_neg_points, visual_obj_id,
                   gaze_mode, gaze_reference_frame):
            """
            Unified object processing function for all prompt types
            """
            global stored_segmentation_data
            
            try:
                if prompt_type == "text":
                    status, gallery_images, detection_data = process_text_prompt(
                        text_prompt, frame_idx, box_thresh, text_thresh,
                        video_key
                    )
                    return status, gallery_images, detection_data, gr.update()
                    
                elif prompt_type == "visual":
                    status, gallery_images, detection_data = process_visual_prompt(
                        visual_frame_idx, visual_pos_points, visual_neg_points, 
                        visual_obj_id, video_key
                    )
                    interactive_image = gallery_images[0] if gallery_images else None
                    return status, gallery_images, detection_data, interactive_image
                    
                elif prompt_type == "gaze":
                    status, gallery_images, detection_data = process_gaze_prompt(
                        gaze_mode, gaze_reference_frame, video_key
                    )
                    return status, gallery_images, detection_data, gr.update()
                    
                else:
                    return "❌ Unknown prompt type", [], None, None
                    
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Processing error: {error_trace}")
                return f"❌ Error: {str(e)}", [], None, None

        def process_text_prompt(text_prompt, frame_idx, box_thresh, text_thresh, video_key):
            
            try:
                # TEXT DETECTION: Use GroundingDINO + SAM2
                if not text_prompt or not text_prompt.strip():
                    return "❌ Please enter object description", [], None
                
                # Handle batch processing for text mode
                if video_key == "__ALL__":
                    # Process all videos for batch mode
                    
                    video_keys = list(tracker.current_video_sessions.keys())

                    if not video_keys:
                        return "❌ No videos loaded", [], None
                    
                    all_previews = []
                    batch_results = []
                    total_objects = []
                    all_scores = []
                    detection_data_dict = {}
                    segmentation_data_dict = {}
                    
                    for vk in video_keys:
                        # Step 1: Object detection with GroundingDINO
                        status, preview, detection_data = tracker.detect_objects_with_text(
                            text_prompt, vk, int(frame_idx), float(box_thresh), float(text_thresh)
                        )

                        
                        if detection_data and len(detection_data.get("boxes", [])) > 0:
                            # Step 2: Segmentation with SAM2 using detected boxes
                            segmentation_status, segmentation_preview, segmentation_data = tracker.segment_detected_objects(
                                vk, int(frame_idx), detection_data
                            )
                            obj_count = len(detection_data.get("boxes", 0))
                            total_objects.append(obj_count)

                            detection_data_dict[vk] = detection_data
                            segmentation_data_dict[vk] = segmentation_data

                            if segmentation_data:
                                
                                # Use segmentation preview if available, otherwise detection preview
                                display_preview = segmentation_preview if segmentation_preview else preview
                                if display_preview:
                                    all_previews.append(display_preview)
                                
                                batch_results.append(f"✅ {vk}: {obj_count} objects detected and segmented")
                            
                            else:
                                batch_results.append(f"❌ {vk}: Segmentation failed")
                            
                        else:
                            batch_results.append(f"❌ {vk}: No objects found matching '{text_prompt}'")

                    
                    stored_segmentation_data = {
                        'type': 'text_batch',
                        'video_keys': video_keys,
                        'detection_data': detection_data_dict,
                        'segmentation_data': segmentation_data_dict,
                        'text_prompt': text_prompt,
                        'frame_idx': frame_idx,
                        'total_objects': total_objects
                        
                        
                    }

                    summary_status = f"✅ Batch text detection completed for {len(video_keys)} videos:\n" + "\n".join(batch_results)
                    return summary_status, all_previews, stored_segmentation_data
                else:
                    # Single video processing
                    vk = video_key if video_key else "video_000"
                    
                    # Step 1: Object detection with GroundingDINO
                    status, preview, detection_data = tracker.detect_objects_with_text(
                        text_prompt, vk, int(frame_idx), float(box_thresh), float(text_thresh)
                    )
                    
                    if not detection_data or len(detection_data.get("boxes", [])) == 0:
                        return f"❌ No objects found matching '{text_prompt}'", [], None
                    
                    # Step 2: Segmentation with SAM2 using detected boxes
                    segmentation_status, segmentation_preview, segmentation_data = tracker.segment_detected_objects(
                        vk, int(frame_idx), detection_data
                    )
                    
                    if segmentation_data:
                    
                        # Combine detection and segmentation data
                        combined_data_for_state = {
                            'type': 'text',
                            'detection_data': {vk: detection_data},
                            'segmentation_data': segmentation_data,
                            'video_key': vk,
                            'frame_idx': frame_idx
                        }
                        
                        stored_segmentation_data = combined_data_for_state
                        
                        # Use segmentation preview if available, otherwise detection preview
                        display_preview = segmentation_preview if segmentation_preview else preview
                        
                        return f"✅ Detection + Segmentation completed!\n{segmentation_status}", [display_preview] if display_preview else [], combined_data_for_state
                    else:
                        return f"❌ Segmentation failed: {segmentation_status}", [], None
                    
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Processing error: {error_trace}")
                return f"❌ Error: {str(e)}", [], None

            
        def process_visual_prompt(frame_idx, pos_points, neg_points, obj_id, video_key):
            """Handle visual point-based detection"""
            if not pos_points:
                return "❌ Please add at least one positive point", [], None
            
            vk = video_key if video_key else "video_000"
            
            status, preview, detection_data = tracker.process_visual_prompt_with_object_id(
                frame_idx, pos_points, neg_points, obj_id, video_key
            )
            
            stored_segmentation_data = {
                'type': 'visual',
                'video_key': vk,
                'frame_idx': frame_idx,
                'detection_data': detection_data
            }
            
            return status, [preview] if preview else [], stored_segmentation_data

        def process_gaze_prompt(gaze_mode, ref_frame, video_key):
            """Handle gaze-based detection with reference vs frame-by-frame modes"""
            
            if gaze_mode == "reference":
                # Reference frame propagation: detect at gaze points in reference frame, then track
                if video_key == "__ALL__":
                    video_keys = list(tracker.current_video_sessions.keys())
                    if not video_keys:
                        return "❌ No videos loaded", [], None
                    
                    all_previews = []
                    batch_results = []
                    
                    for vk in video_keys:
                        status, previews = tracker.segment_gaze_reference_propagation(
                            vk, int(ref_frame)
                        )
                        if previews:
                            all_previews.extend(previews[:2])
                        batch_results.append(status)
                    
                    stored_segmentation_data = {
                        'type': 'gaze_reference_batch',
                        'video_keys': video_keys,
                        'reference_frame': ref_frame,
                        'mode': 'reference'
                    }
                    
                    summary_status = f"✅ Batch gaze reference propagation completed for {len(video_keys)} videos:\n" + "\n".join(batch_results)
                    return summary_status, all_previews, stored_segmentation_data
                
                else:
                    # Single video reference propagation
                    vk = video_key if video_key else "video_000"
                    
                    status, previews = tracker.segment_gaze_reference_propagation(
                        vk, int(ref_frame), 
                    )
                    
                    stored_segmentation_data = {
                        'type': 'gaze_reference',
                        'video_key': vk,
                        'reference_frame': ref_frame,
                        'mode': 'reference'
                    }
                    
                    return status, previews or [], stored_segmentation_data
            
            else:  # frame_by_frame mode
                # Frame-by-frame: detect objects at gaze points independently for each frame
                if video_key == "__ALL__":
                    video_keys = list(tracker.current_video_sessions.keys())
                    if not video_keys:
                        return "❌ No videos loaded", [], None
                    
                    all_previews = []
                    batch_results = []
                    
                    for vk in video_keys:
                        status, previews = tracker.segment_gaze_per_frame(
                            vk
                        )
                        if previews:
                            all_previews.extend(previews[:2])
                        batch_results.append(status)
                    
                    stored_segmentation_data = {
                        'type': 'gaze_frame_by_frame_batch',
                        'video_keys': video_keys,
                        'mode': 'frame_by_frame'
                    }
                    
                    summary_status = f"✅ Batch frame-by-frame gaze detection completed for {len(video_keys)} videos:\n" + "\n".join(batch_results)
                    return summary_status, all_previews, stored_segmentation_data
                
                else:
                    # Single video frame-by-frame
                    vk = video_key if video_key else "video_000"
                    
                    status, previews = tracker.segment_gaze_per_frame(
                        vk
                    )
                    
                    stored_segmentation_data = {
                        'type': 'gaze_frame_by_frame',
                        'video_key': vk,
                        'mode': 'frame_by_frame'
                    }
                    
                    return status, previews or [], stored_segmentation_data

       
        def execute_tracking_with_segmentation(segmentation_data, prompt_type):
            """Execute tracking with support for frame-by-frame gaze mode and batch processing"""
            global stored_segmentation_data
            
            try:
                if not tracker:
                    return "❌ Tracker not initialized. Please load a video first."
                
                if not segmentation_data:
                    if stored_segmentation_data:
                        segmentation_data = stored_segmentation_data
                    else:
                        return "❌ No segmentation data available. Please process objects first."
                
                if prompt_type == "gaze":
                    # Handle batch gaze tracking
                    if segmentation_data.get('type') == 'gaze_batch':
                        video_keys = segmentation_data.get('video_keys', [])
                        mode = segmentation_data.get('mode', 'frame_by_frame')
                        
                        if mode == 'frame_by_frame':
                            # Frame-by-frame mode - already processed, just return success
                            return f"✅ Batch frame-by-frame gaze tracking completed for {len(video_keys)} videos!"
        
                    elif segmentation_data.get('type') == 'gaze_frame_by_frame':
                        # Single video frame-by-frame gaze tracking
                        video_key = segmentation_data.get('video_key', 'video_000')
                        return f"✅ Frame-by-frame gaze tracking completed!"
                    
                elif prompt_type == "text":
                    # Handle batch text tracking
                    if segmentation_data.get('type') == 'text_batch':
                        video_keys = segmentation_data['detection_data'].keys()
                        text_prompt = segmentation_data.get('text_prompt', '')
                        frame_idx = segmentation_data.get('frame_idx', 0)
                        total_objects = segmentation_data.get('total_objects', [])

                        
                        batch_results = []
                        
                        for (video_key, obj_count) in zip(video_keys, total_objects):
                            
                            if obj_count > 0:
                                status = tracker.run_sam2_tracking(
                                    video_key, frame_idx, segmentation_data, prompt_type
                                )
                                
                                batch_results.append(f"✅ {video_key}: {obj_count} objects tracked.")
                            else:
                                batch_results.append(f"❌ {video_key}: No objects detected to track.")
                            
                        return str(status) + "\n" + "\n".join(batch_results)
                        # return f"✅ Batch text tracking completed:\n" + "\n".join(batch_results)
                    
                    else:
                        # Single video text-based tracking 
                        text_segmentation_data = segmentation_data.get('segmentation_data')
                        
                        video_key = segmentation_data.get('video_key', 'video_000')
                        text_detection_data = segmentation_data['detection_data'][video_key]
                        frame_idx = segmentation_data.get('frame_idx', 0)
                        obj_count = len(text_detection_data.get('boxes', [])) if text_detection_data else 0
                        
                        if not text_segmentation_data:
                            return "❌ No text segmentation data found."
                        
                        if obj_count > 0:
                            # Execute text-based tracking
                            status = tracker.run_sam2_tracking(
                                video_key, int(frame_idx), segmentation_data, prompt_type
                            )
                            return f"✅ Text-based tracking completed!\n🎯 {obj_count} detected objects tracked\n{status}"

                        else:
                            return "❌ No objects detected to track."
                
                elif prompt_type == "visual":
                    visual_detection_data = segmentation_data.get('detection_data')
                    video_key = segmentation_data.get('video_key', 'video_000')
                    frame_idx = segmentation_data.get('frame_idx', 0)
                    
                    if not visual_detection_data:
                        return "❌ No segmentation data found."
                    
                    # Execute text-based tracking
                    status = tracker.run_sam2_tracking(
                        video_key, int(frame_idx), visual_detection_data, prompt_type
                    )
                    visual_detection_data = segmentation_data.get('detection_data')
                    obj_count = len(visual_detection_data.get('boxes', [])) if visual_detection_data else 0
                    return f"✅ Tracking completed!\n🎯 {obj_count} detected objects tracked\n{status}"


                else:
                    return f"❌ Unknown prompt type: {prompt_type}"
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                logger.error(f"Tracking execution error: {error_details}")
                return f"❌ Tracking failed: {str(e)}\n\nPlease check that:\n1. Video is loaded\n2. Objects are processed\n3. SAM2 model is initialized"

        def export_with_frame_by_frame_support(output_dir, export_formats, gaze_exports, 
                                     include_overlay, video_key, seg_data):
            """ Export that handles frame-by-frame gaze results, batch processing, and advanced features"""
            try:
                # Handle batch export or single video export
                if seg_data and seg_data.get('type') in ['gaze_batch', 'text_batch']:
                    # Batch export - process all videos
                    video_keys = seg_data.get('video_keys', [])
                    if not video_keys:
                        return "❌ No videos to export"
                    
                    export_results = []
                    for vk in video_keys:
                        try:
                            status = tracker.save_results_with_gaze(
                                output_dir,  
                                save_video="video" in export_formats,
                                save_csv="csv" in export_formats,
                                save_elan="elan" in export_formats,
                                video_key=vk,
                                include_gaze_overlay=include_overlay,
                                save_frames=True,  # Don't save individual frames for batch
                                save_confidence=True
                            )
                            video_name = tracker.current_video_sessions.get(vk, {}).get('video_name', vk)
                            export_results.append(f"✅ {video_name}: exported successfully")
                        except Exception as e:
                            video_name = tracker.current_video_sessions.get(vk, {}).get('video_name', vk)
                            export_results.append(f"❌ {video_name}: export failed - {str(e)}")
                    
                    # Create combined results summary
                    combined_status = f"✅ Batch export completed for {len(video_keys)} videos:\n" + "\n".join(export_results)
                    
                    # Add combined CSV if multiple videos
                    if len(video_keys) > 1 and "csv" in export_formats:
                        combined_status += "\n✅ Combined CSV created"
                    
                else:
                    # Single video export or fallback
                    export_video_key = None if video_key == "__ALL__" else video_key
                    
                    status = tracker.save_results_with_gaze(
                        output_dir, 
                        save_video="video" in export_formats,
                        save_csv="csv" in export_formats,
                        save_elan="elan" in export_formats,
                        video_key=export_video_key,
                        include_gaze_overlay=include_overlay,
                        save_frames=True,
                        save_confidence=True
                    )
                    combined_status = status
                
                return combined_status
                
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Export error: {error_trace}")
                return f"❌ Export failed: {str(e)}"
                
        
        # Wire up event handlers

        # Toggle input visibility when switching between single file and directory modes
        input_method.change(
            fn=toggle_input_visibility,
            inputs=[input_method],
            outputs=[video_file, video_dir, gaze_csv_file, gaze_csv_dir]
        )
        
        prompt_type.change(
            fn=toggle_prompt_visibility,
            inputs=[prompt_type],
            outputs=[textual_group, visual_group, gaze_group, gallery_view, interactive_view]
        )

        
        # init_btn.click(
        #     fn=tracker.initialize_models,
        #     outputs=[init_status]
        # )
        
        # Process button with directory support

        process_btn.click(
            fn=handle_video_and_gaze_processing,
            inputs=[video_file, video_dir, input_method, gaze_csv_file, gaze_csv_dir],
            outputs=[video_status,  video_selection_row, video_selector]
        ).then(
            fn=tracker.initialize_models,
            outputs=[]
        ).then(
            # ALSO update mask correction video selector
            fn=update_mask_correction_video_selector,
            inputs=[],
            outputs=[mask_correction_video_selector]
        )

        # Event handlers - SINGLE SET, NO DUPLICATES

        # Toggle gaze mode visibility
        gaze_mode.change(
            fn=toggle_gaze_mode_visibility,
            inputs=[gaze_mode],
            outputs=[gaze_reference_group, gaze_framewise_group]
        )
        
        detect_btn.click(
            fn=process_objects,
            inputs=[
                # Text inputs
                text_prompt, frame_idx, box_threshold, text_threshold,
                # Common inputs
                video_selector, prompt_type,
                # Visual inputs
                visual_frame_idx, visual_positive_points, visual_negative_points, visual_current_object_id,
                # Gaze inputs
                gaze_mode, gaze_reference_frame
            ],
            outputs=[
                detection_status,           # status message
                results_preview_gallery,    # gallery (used for text/gaze)
                detection_data,            # detection data
                results_interactive_image
            ]
        )

        # approve_btn.click(
        #     fn=execute_tracking_with_segmentation,
        #     inputs=[detection_data, prompt_type],
        #     outputs=[detection_status]
        # )

        approve_btn.click(
            fn=execute_tracking_with_segmentation,
            inputs=[detection_data, prompt_type],
            outputs=[detection_status]
        ).then(
            # Refresh the mask correction video selector
            fn=update_mask_correction_video_selector,
            inputs=[],
            outputs=[mask_correction_video_selector]
        ).then(
            # Update object choices for the first video
            fn=lambda: update_object_choices(
                tracker.get_video_list()[0] if tracker.get_video_list() else None
            ),
            inputs=[],
            outputs=[obj_choice]
        )
        refine_btn.click(
            fn=lambda: gr.update(selected="mask_correction"),  
            outputs=main_tabs
        )

        # Object dropdown change handler for mask correction
        obj_choice.change(
            fn=lambda vk, oid: switch_object(vk, oid) if oid is not None else ("📋 No objects available", None, create_confidence_plot()),
            inputs=[mask_correction_video_selector, obj_choice],
            outputs=[correction_status, video_frame_image, confidence_plot]
        )

        # Event handlers
        load_visual_frame_btn.click(
            fn=load_frame_for_visual_annotation,
            inputs=[visual_frame_idx],
            outputs=[results_interactive_image, visual_points_info]  # Load into interactive image on right
        )

        # Connect click events to the interactive results image
        results_interactive_image.select(
            fn=handle_image_click,
            inputs=[results_interactive_image, visual_correction_mode, visual_positive_points, visual_negative_points, visual_frame_idx],
            outputs=[results_interactive_image, visual_positive_points, visual_negative_points, visual_points_info]
        )

        frame_timeline.change(
            fn=update_frame_from_slider_with_video,
            inputs=[frame_timeline, mask_correction_video_selector],
            outputs=[video_frame_image, correction_status, positive_points, negative_points, points_info]
        ).then(
            fn=redraw_overlay,
            inputs=[mask_correction_video_selector, frame_timeline, positive_points, negative_points, obj_choice],
            outputs=[video_frame_image]
        )
    
        
        clear_visual_points_btn.click(
            fn=clear_visual_points,
            inputs=[visual_frame_idx],
            outputs=[visual_positive_points, visual_negative_points, visual_points_info]
        )
 
        export_btn.click(
            fn=export_with_frame_by_frame_support,
            inputs=[output_dir, export_formats, gaze_exports, 
                include_gaze_overlay, video_selector, current_segmentation_data],
            outputs=[export_status]
        )
    
    return demo