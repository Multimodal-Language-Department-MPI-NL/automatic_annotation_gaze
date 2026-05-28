"""
ELAN (.eaf) Export Module for Object Tracking Validation
Generates XML-based ELAN Annotation Format files for validation against human annotations

Tier Structure:
- gaze_intersection: Time-alignable tier with gaze-on-object segments (independent tier)
  - Each annotation represents a time period when gaze intersected with a specific object
  - Value: Object label (e.g., "person", "laptop", "cup")
  - Multiple overlapping annotations allowed (multiple objects gazed at simultaneously)
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
import os

logger = logging.getLogger(__name__)


class ELANExporter:
    """
    Export gaze-on-object tracking data to ELAN Annotation Format (.eaf) for validation
    
    Creates a single 'gaze_intersection' tier showing when gaze intersected with tracked objects.
    Multiple overlapping annotations represent simultaneous gaze on different objects.
    """
    
    def __init__(self, video_name: str, video_path: str, 
                 fps: float = 25, 
                 duration_ms: int = 0,
                 author: str = "SAM2_Tracker"):
        """
        Initialize ELAN exporter
        
        Args:
            video_name: Name of the video file
            video_path: Path to the video file
            fps: Frames per second
            duration_ms: Total video duration in milliseconds
            author: Author of annotations
        """
        self.video_name = video_name
        self.video_path = video_path
        self.fps = fps
        self.duration_ms = duration_ms
        self.author = author
        self.time_slots = {}  # {slot_id: time_ms}
        self.slot_counter = 1
        self.annotation_counter = 1
        
        # Initialize XML structure
        self.root = ET.Element('ANNOTATION_DOCUMENT', {
            'AUTHOR': author,
            'DATE': datetime.now().isoformat(),
            'FORMAT': '3.0',
            'VERSION': '3.0',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:noNamespaceSchemaLocation': 'http://www.mpi.nl/tools/elan/EAFv3.0.xsd'
        })
        
        # Add header
        self._add_header()
        
        # Time order element (will be populated)
        self.time_order = ET.SubElement(self.root, 'TIME_ORDER')
        
        # Store annotations for later processing - only gaze intersections now
        self.gaze_intersection_annotations = []  # List of (start_ms, end_ms, object_label)
    
    def _add_header(self):
        """Add ELAN header with media file information"""
        header = ET.SubElement(self.root, 'HEADER', {
            'MEDIA_FILE': '',
            'TIME_UNITS': 'milliseconds'
        })
        
        # Media descriptor
        media_url = self.video_path.replace('\\', '/')
        if not media_url.startswith('file:///'):
            media_url = f'file:///{media_url}'
        
        ET.SubElement(header, 'MEDIA_DESCRIPTOR', {
            'MEDIA_URL': media_url,
            'MIME_TYPE': 'video/mp4',
            'RELATIVE_MEDIA_URL': f'./{os.path.basename(self.video_path)}'
        })
        
        # Properties
        ET.SubElement(header, 'PROPERTY', {
            'NAME': 'URN'
        }).text = f'urn:nl-mpi-tools-elan-eaf:{self.video_name}'
        
        ET.SubElement(header, 'PROPERTY', {
            'NAME': 'lastUsedAnnotationId'
        }).text = '0'
    
    def frame_to_ms(self, frame_idx: int) -> int:
        """Convert frame index to milliseconds"""
        return int((frame_idx / self.fps) * 1000)
    
    def _get_or_create_time_slot(self, time_ms: int) -> str:
        """Get existing time slot ID or create new one"""
        # Check if this time already has a slot
        for slot_id, existing_time in self.time_slots.items():
            if existing_time == time_ms:
                return slot_id
        
        # Create new slot
        slot_id = f'ts{self.slot_counter}'
        self.time_slots[slot_id] = time_ms
        self.slot_counter += 1
        return slot_id
    
    def _get_annotation_id(self) -> str:
        """Get next annotation ID"""
        ann_id = f'a{self.annotation_counter}'
        self.annotation_counter += 1
        return ann_id
    
    def add_gaze_intersection(self, start_frame: int, end_frame: int, object_label: str):
        """
        Add a gaze intersection segment
        
        Args:
            start_frame: Starting frame of gaze intersection
            end_frame: Ending frame of gaze intersection (inclusive)
            object_label: Label of the object being gazed at
        """
        # Validate inputs
        if object_label is None or str(object_label).strip() == "":
            logger.warning(f"Skipping annotation with empty label at frames {start_frame}-{end_frame}")
            return
        
        if start_frame < 0 or end_frame < start_frame:
            logger.warning(f"Invalid frame range: {start_frame}-{end_frame}, skipping")
            return
        
        start_ms = self.frame_to_ms(start_frame)
        # For end frame: calculate end of that frame (not beginning of next frame)
        # Add one frame duration to make it inclusive
        frame_duration_ms = int(1000 / self.fps)
        end_ms = self.frame_to_ms(end_frame) + frame_duration_ms
        
        # Ensure label is a valid string
        label_str = str(object_label).strip()
        
        self.gaze_intersection_annotations.append({
            'start_ms': start_ms,
            'end_ms': end_ms,
            'label': label_str
        })
        
        logger.debug(f"Added gaze intersection: '{label_str}' from frame {start_frame} to {end_frame} ({start_ms}ms to {end_ms}ms)")
    
    def _build_time_order(self):
        """Build TIME_ORDER element with all unique time slots"""
        # Clear any existing time slots
        self.time_order.clear()
        
        # Sort time slots by time value
        sorted_slots = sorted(self.time_slots.items(), key=lambda x: x[1])
        
        for slot_id, time_ms in sorted_slots:
            ET.SubElement(self.time_order, 'TIME_SLOT', {
                'TIME_SLOT_ID': slot_id,
                'TIME_VALUE': str(time_ms)
            })
    
    def _build_tiers(self):
        """Build tier elements with annotations"""
        # Create gaze_intersection tier (independent, time-alignable)
        if self.gaze_intersection_annotations:
            gaze_tier = ET.SubElement(self.root, 'TIER', {
                'LINGUISTIC_TYPE_REF': 'default-lt',
                'TIER_ID': 'gaze_intersection'
            })
            
            annotations_added = 0
            for gaze_data in self.gaze_intersection_annotations:
                # Double-check label is not empty
                label = gaze_data.get('label', '').strip()
                if not label:
                    logger.warning(f"Skipping annotation with empty label at {gaze_data['start_ms']}-{gaze_data['end_ms']}ms")
                    continue
                
                # Create time slots for THIS annotation's actual start/end time
                start_slot = self._get_or_create_time_slot(gaze_data['start_ms'])
                end_slot = self._get_or_create_time_slot(gaze_data['end_ms'])
                
                annotation = ET.SubElement(gaze_tier, 'ANNOTATION')
                alignable_ann = ET.SubElement(annotation, 'ALIGNABLE_ANNOTATION', {
                    'ANNOTATION_ID': self._get_annotation_id(),
                    'TIME_SLOT_REF1': start_slot,
                    'TIME_SLOT_REF2': end_slot
                })
                # Set the annotation value to the object label
                ET.SubElement(alignable_ann, 'ANNOTATION_VALUE').text = label
                annotations_added += 1
            
            logger.info(f"Built gaze_intersection tier with {annotations_added} annotations")
        else:
            logger.warning("No gaze intersection annotations to export")
    
    def _build_linguistic_types(self):
        """Build linguistic type definitions"""
        # Default type (time-alignable) - only type we need
        ET.SubElement(self.root, 'LINGUISTIC_TYPE', {
            'GRAPHIC_REFERENCES': 'false',
            'LINGUISTIC_TYPE_ID': 'default-lt',
            'TIME_ALIGNABLE': 'true'
        })
    
    def _build_constraints(self):
        """Build constraint definitions"""
        # No constraints needed for independent tier
        pass
    
    def save(self, output_path: str) -> bool:
        """
        Save ELAN file to disk
        
        Args:
            output_path: Path where to save the .eaf file
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Saving ELAN file with {len(self.gaze_intersection_annotations)} total annotations")
            
            # Build all components IN THE RIGHT ORDER
            # 1. Build tiers first (this creates time slots)
            self._build_tiers()
            # 2. Now build time order (after time slots are created)
            self._build_time_order()
            # 3. Build linguistic types and constraints
            self._build_linguistic_types()
            self._build_constraints()
            
            # Convert to pretty XML
            xml_str = ET.tostring(self.root, encoding='utf-8')
            dom = minidom.parseString(xml_str)
            pretty_xml = dom.toprettyxml(indent='    ', encoding='utf-8')
            
            # Write to file
            with open(output_path, 'wb') as f:
                f.write(pretty_xml)
            
            logger.info(f"✅ ELAN file saved: {output_path}")
            logger.info(f"   - {len(self.gaze_intersection_annotations)} gaze intersection segments")
            logger.info(f"   - {len(self.time_slots)} time slots")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving ELAN file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def merge_consecutive_frames(frame_data: List[Tuple[int, int, str]]) -> List[Dict]:
    """
    Merge consecutive frames for the same object into segments
    
    Args:
        frame_data: List of (frame_idx, object_id, object_label) tuples
    
    Returns:
        List of segment dicts with start_frame, end_frame, object_id, object_label
    """
    if not frame_data:
        return []
    
    # Sort by object_id, then frame
    sorted_data = sorted(frame_data, key=lambda x: (x[1], x[0]))
    
    segments = []
    current_segment = None
    
    for frame_idx, obj_id, obj_label in sorted_data:
        # Validate label is not empty
        if obj_label is None or str(obj_label).strip() == "":
            logger.warning(f"Skipping frame {frame_idx} object {obj_id} with empty label")
            continue
        
        if current_segment is None:
            # Start new segment
            current_segment = {
                'start_frame': frame_idx,
                'end_frame': frame_idx,
                'object_id': obj_id,
                'object_label': str(obj_label).strip()
            }
        elif (obj_id == current_segment['object_id'] and 
              frame_idx == current_segment['end_frame'] + 1):
            # Continue current segment
            current_segment['end_frame'] = frame_idx
        else:
            # Save current segment and start new one
            segments.append(current_segment)
            current_segment = {
                'start_frame': frame_idx,
                'end_frame': frame_idx,
                'object_id': obj_id,
                'object_label': str(obj_label).strip()
            }
    
    # Don't forget last segment
    if current_segment is not None:
        segments.append(current_segment)
    
    return segments