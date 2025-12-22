"""
VizForge Video Export Engine

Export charts as animated videos (MP4, WebM, GIF).
Perfect for presentations, social media, and reports!

Features:
- MP4 export (H.264 encoding)
- WebM export (VP9 encoding)
- GIF export (optimized)
- Custom frame rates
- Smooth transitions
- Progress callbacks
"""

from .video_exporter import VideoExporter, export_video, VideoConfig, VideoFormat
from .animation_engine import AnimationEngine, AnimationType
from .frame_generator import FrameGenerator

__all__ = [
    # Main entry point
    'export_video',

    # Core classes
    'VideoExporter',
    'VideoConfig',
    'VideoFormat',
    'AnimationEngine',
    'AnimationType',
    'FrameGenerator',
]
