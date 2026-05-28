import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
import os
from typing import List, Tuple, Optional, Dict
import logging

class GazeProcessor:
    def __init__(self):
        # Create logger instance for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.gaze_data = None
        self.frame_gaze_cache = {}  # Cache for frame-specific gaze points
        
    def load_gaze_csv(self, csv_file_path, frame_col="frame", x_col="x", y_col="y"):
        """
        Load gaze data from CSV with flexible column mapping
        Expected format: frame, x, y, confidence (optional)
        """
        try:
            # Read CSV
            if hasattr(csv_file_path, 'name'):  # File upload object
                df = pd.read_csv(csv_file_path.name)
            else:
                df = pd.read_csv(csv_file_path)
            
            # Validate required columns
            required_cols = [frame_col, x_col, y_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
            
            # Rename columns to standard names
            column_mapping = {
                frame_col: 'frame',
                x_col: 'x',
                y_col: 'y'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns are numeric
            for col in ['frame', 'x', 'y']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN values in required columns
            initial_rows = len(df)
            df = df.dropna(subset=['frame', 'x', 'y'])
            final_rows = len(df)
            
            if final_rows == 0:
                return False, "No valid gaze data after cleaning"
            
            # Sort by frame and ensure frame indices are integers
            df['frame'] = df['frame'].astype(int)
            df = df.sort_values('frame').reset_index(drop=True)
            
            # Add confidence column if not present
            if 'confidence' not in df.columns:
                df['confidence'] = 1.0
            
            self.gaze_data = df
            self.frame_gaze_cache = {}  # Reset cache

            # Create frame-wise index for faster access
            self._create_frame_index()
            
            dropped_rows = initial_rows - final_rows
            success_msg = f"Loaded {final_rows:,} gaze points from {len(df['frame'].unique())} frames"
            if dropped_rows > 0:
                success_msg += f" (dropped {dropped_rows} invalid rows)"
            
            self.logger.info(success_msg)
            return True, success_msg
            
        except Exception as e:
            self.logger.error(f"Error loading gaze CSV: {e}")
            return False, f"Error loading CSV: {str(e)}"
    
    def load_gaze_directory(self, gaze_dir_path, frame_col="frame", x_col="x", y_col="y"):
        """
        Load gaze data from a directory of CSV files using same filename matching
        (video.mp4 -> video.csv)
        
        Args:
            gaze_dir_path: Path to directory containing CSV files
            frame_col, x_col, y_col: Column names for frame, x, y coordinates
            
        Returns:
            (success: bool, message: str)
        """
        try:
            import os
            from pathlib import Path
            
            if not os.path.exists(gaze_dir_path) or not os.path.isdir(gaze_dir_path):
                return False, f"Gaze directory does not exist: {gaze_dir_path}"
            
            # Find all CSV files in directory
            csv_files = []
            for file in os.listdir(gaze_dir_path):
                if file.lower().endswith('.csv'):
                    csv_files.append(os.path.join(gaze_dir_path, file))
            
            if not csv_files:
                return False, f"No CSV files found in directory: {gaze_dir_path}"
            
            # Load and combine all CSV files
            all_dataframes = []
            video_file_mapping = {}
            files_processed = 0
            
            for csv_file in csv_files:
                try:
                    # Extract base name for matching (video.csv -> video)
                    csv_basename = Path(csv_file).stem
                    video_basename = csv_basename  # Same name matching
                    
                    # Load this CSV file
                    df = pd.read_csv(csv_file)
                    
                    # Validate required columns
                    required_cols = [frame_col, x_col, y_col]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        self.logger.warning(f"CSV {csv_file} missing columns {missing_cols}, skipping")
                        continue
                    
                    # Rename columns to standard names
                    column_mapping = {
                        frame_col: 'frame',
                        x_col: 'x',
                        y_col: 'y'
                    }
                    
                    df = df.rename(columns=column_mapping)
                    
                    # Ensure required columns are numeric and clean data
                    for col in ['frame', 'x', 'y']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Remove rows with NaN values
                    initial_rows = len(df)
                    df = df.dropna(subset=['frame', 'x', 'y'])
                    final_rows = len(df)
                    
                    if final_rows == 0:
                        self.logger.warning(f"No valid data in {csv_file} after cleaning, skipping")
                        continue
                    
                    # Add video identifier column for multi-video tracking
                    df['video_basename'] = video_basename
                    df['csv_source'] = csv_file
                    
                    # Ensure frame indices are integers
                    df['frame'] = df['frame'].astype(int)
                    
                    # Add default columns if missing
                    if 'confidence' not in df.columns:
                        df['confidence'] = 1.0
                    
                    all_dataframes.append(df)
                    video_file_mapping[video_basename] = csv_file
                    files_processed += 1
                    
                    self.logger.info(f"Loaded {final_rows} gaze points from {csv_file} (video: {video_basename})")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {csv_file}: {e}")
                    continue
            
            if not all_dataframes:
                return False, f"No valid CSV files could be processed from {gaze_dir_path}"
            
            # Combine all dataframes
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            combined_df = combined_df.sort_values(['video_basename', 'frame']).reset_index(drop=True)
            
            # Store the combined data
            self.gaze_data = combined_df
            self.frame_gaze_cache = {}  # Reset cache
            self.video_file_mapping = video_file_mapping  # Store mapping for reference
            
            # Create frame-wise index for faster access
            self._create_frame_index()
            
            total_points = len(combined_df)
            total_frames = len(combined_df['frame'].unique())
            videos_count = len(combined_df['video_basename'].unique())
            
            success_msg = (f"Loaded {total_points:,} gaze points from {files_processed} CSV files "
                         f"covering {videos_count} videos across {total_frames} frames")
            
            self.logger.info(success_msg)
            return True, success_msg
            
        except Exception as e:
            self.logger.error(f"Error loading gaze directory: {e}")
            return False, f"Error loading gaze directory: {str(e)}"
    
    def _create_frame_index(self):
        """Create an index for fast frame-based lookups"""
        if self.gaze_data is None:
            return
        
        self.frame_index = {}
        for frame_num in self.gaze_data['frame'].unique():
            frame_data = self.gaze_data[self.gaze_data['frame'] == frame_num]
            self.frame_index[frame_num] = frame_data.index.tolist()
    
    def has_gaze_data(self):
        """Check if gaze data is loaded"""
        return self.gaze_data is not None and len(self.gaze_data) > 0
    
    def set_processing_options(self, **options):
        """Update processing options"""
        self.processing_options.update(options)
        self.frame_gaze_cache = {}  # Clear cache when options change

    def get_frame_gaze_points(self, frame_idx: int, 
                            min_confidence: float = 0.0, video_basename: str = None) -> List[Tuple[float, float]]:
        """
        Get gaze points for a specific frame, optionally filtered by video
        
        Args:
            frame_idx: Frame number to get gaze points for
            min_confidence: Minimum confidence threshold
            video_basename: Optional video basename to filter by (for multi-video datasets)
        """

        if self.gaze_data is None:
            return []
        
        # Filter by frame
        data = self.gaze_data[self.gaze_data['frame'] == frame_idx]

        # Filter by video if specified (for directory-loaded data)
        if video_basename and 'video_basename' in data.columns:
            data = data[data['video_basename'] == video_basename]
        
        # Filter by confidence
        if 'confidence' in data.columns:
            data = data[data['confidence'] >= float(min_confidence)]
        
        pts = data[['x','y']].values
        
        return [tuple(p) for p in pts]
    
    def overlay_gaze_on_image(self, image, frame_idx, show_points=True, point_size=3, video_basename=None):
        """
        Overlay gaze points on an image for the specified frame
        
        Args:
            image: PIL Image to overlay gaze on
            frame_idx: Frame number
            show_points: Whether to show gaze points
            point_size: Size of gaze points
            video_basename: Optional video basename for multi-video datasets
        """
        if not self.has_gaze_data():
            return image
        
        try:
            # Get gaze points for this frame (optionally filtered by video)
            gaze_points = self.get_frame_gaze_points(frame_idx, video_basename=video_basename)
            
            if not gaze_points:
                return image
            
            # Create a copy of the image
            overlay_image = image.copy()
            draw = ImageDraw.Draw(overlay_image)
            
            # Draw gaze points
            for i, (x, y) in enumerate(gaze_points):
                x, y = int(x), int(y)
                
                if show_points:
                    # Draw point
                    draw.ellipse([x-point_size, y-point_size, x+point_size, y+point_size], 
                               fill='red', outline='red', width=10)
            
            return overlay_image
            
        except Exception as e:
            self.logger.error(f"Error overlaying gaze on image: {e}")
            return image
    
    def export_gaze_analysis(self, output_dir: str, export_types: List[str] = None) -> str:
        """
        Export gaze analysis results to various formats
        
        Args:
            output_dir: Directory to save files
            export_types: List of export types to include
            
        Returns:
            Status message with exported files
        """
        if export_types is None:
            export_types = ["trajectories", "heatmaps", "gaze_object", "stats"]
            
        output_files = []
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Export raw gaze data
            if self.gaze_data is not None:
                gaze_path = os.path.join(output_dir, "gaze_data.csv")
                self.gaze_data.to_csv(gaze_path, index=False)
                output_files.append("gaze_data.csv")
            
            if output_files:
                return f"Exported {len(output_files)} gaze analysis files: {', '.join(output_files)}"
            else:
                return "No gaze analysis files exported"
                
        except Exception as e:
            self.logger.error(f"Error exporting gaze analysis: {e}")
            return f"Export failed: {str(e)}"