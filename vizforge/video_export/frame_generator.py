"""
Frame generator for video exports.

Manages frame generation, interpolation, and caching for video creation.
"""

import os
import io
from typing import List, Optional, Callable, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image


class FrameGenerator:
    """Generate frames for video export."""

    def __init__(
        self,
        chart,
        width: int = 1920,
        height: int = 1080,
        dpi: int = 100
    ):
        """
        Initialize frame generator.

        Args:
            chart: VizForge chart object
            width: Frame width in pixels
            height: Frame height in pixels
            dpi: Dots per inch for rendering
        """
        self.chart = chart
        self.width = width
        self.height = height
        self.dpi = dpi
        self.frames = []

    def generate_from_data_frames(
        self,
        data_frames: List[pd.DataFrame],
        progress_callback: Optional[Callable] = None
    ) -> List[Image.Image]:
        """
        Generate frames from list of DataFrames.

        Args:
            data_frames: List of DataFrames to render
            progress_callback: Optional callback for progress updates

        Returns:
            List of PIL Image objects
        """
        frames = []
        total = len(data_frames)

        for i, df in enumerate(data_frames):
            if progress_callback:
                progress_callback(i / total)

            # Create chart with new data
            frame_chart = self._create_chart_with_data(df)

            # Render to image
            img = self._render_chart_to_image(frame_chart)
            frames.append(img)

        if progress_callback:
            progress_callback(1.0)

        self.frames = frames
        return frames

    def generate_interpolated_frames(
        self,
        start_df: pd.DataFrame,
        end_df: pd.DataFrame,
        n_frames: int,
        easing_function: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Image.Image]:
        """
        Generate interpolated frames between two data states.

        Args:
            start_df: Starting DataFrame
            end_df: Ending DataFrame
            n_frames: Number of frames to generate
            easing_function: Optional easing function for interpolation
            progress_callback: Optional callback for progress

        Returns:
            List of PIL Image objects
        """
        frames = []

        # Interpolate data
        interpolated_data = self._interpolate_dataframes(
            start_df, end_df, n_frames, easing_function
        )

        # Generate frames
        for i, df in enumerate(interpolated_data):
            if progress_callback:
                progress_callback(i / len(interpolated_data))

            frame_chart = self._create_chart_with_data(df)
            img = self._render_chart_to_image(frame_chart)
            frames.append(img)

        if progress_callback:
            progress_callback(1.0)

        self.frames = frames
        return frames

    def save_frames(
        self,
        output_dir: str,
        prefix: str = "frame",
        format: str = "png"
    ) -> List[str]:
        """
        Save all frames to directory.

        Args:
            output_dir: Directory to save frames
            prefix: Filename prefix
            format: Image format (png, jpg, etc.)

        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        paths = []
        for i, frame in enumerate(self.frames):
            filename = f"{prefix}_{i:05d}.{format}"
            filepath = os.path.join(output_dir, filename)
            frame.save(filepath, format=format.upper())
            paths.append(filepath)

        return paths

    def get_frame_pattern(self, output_dir: str, prefix: str = "frame") -> str:
        """
        Get ffmpeg-compatible frame pattern.

        Args:
            output_dir: Directory containing frames
            prefix: Filename prefix

        Returns:
            Frame pattern string (e.g., "frames/frame_%05d.png")
        """
        return os.path.join(output_dir, f"{prefix}_%05d.png")

    def _create_chart_with_data(self, df: pd.DataFrame):
        """Create new chart instance with given data."""
        # Get chart configuration
        chart_class = self.chart.__class__
        config = self.chart._get_config() if hasattr(self.chart, '_get_config') else {}

        # Create new chart with updated data
        return chart_class(data=df, **config)

    def _render_chart_to_image(self, chart) -> Image.Image:
        """Render chart to PIL Image."""
        # Export chart as PNG bytes
        img_bytes = chart.fig.to_image(
            format="png",
            width=self.width,
            height=self.height
        )

        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_bytes))
        return img

    def _interpolate_dataframes(
        self,
        start_df: pd.DataFrame,
        end_df: pd.DataFrame,
        n_frames: int,
        easing_function: Optional[Callable] = None
    ) -> List[pd.DataFrame]:
        """
        Interpolate between two DataFrames.

        Args:
            start_df: Starting DataFrame
            end_df: Ending DataFrame
            n_frames: Number of frames
            easing_function: Optional easing function

        Returns:
            List of interpolated DataFrames
        """
        if easing_function is None:
            easing_function = lambda t: t  # Linear

        # Ensure same structure
        if not start_df.columns.equals(end_df.columns):
            raise ValueError("DataFrames must have same columns")

        interpolated = []

        for i in range(n_frames):
            t = i / (n_frames - 1) if n_frames > 1 else 1.0
            eased_t = easing_function(t)

            # Interpolate each numeric column
            df_interpolated = start_df.copy()

            for col in start_df.columns:
                if pd.api.types.is_numeric_dtype(start_df[col]):
                    start_values = start_df[col].values
                    end_values = end_df[col].values
                    interpolated_values = start_values + (end_values - start_values) * eased_t
                    df_interpolated[col] = interpolated_values

            interpolated.append(df_interpolated)

        return interpolated

    def optimize_frames(
        self,
        quality: int = 85,
        resize_factor: float = 1.0
    ) -> None:
        """
        Optimize frames for smaller file size.

        Args:
            quality: JPEG quality (1-100)
            resize_factor: Factor to resize frames (e.g., 0.5 for half size)
        """
        optimized = []

        for frame in self.frames:
            # Resize if needed
            if resize_factor != 1.0:
                new_size = (
                    int(frame.width * resize_factor),
                    int(frame.height * resize_factor)
                )
                frame = frame.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to RGB if needed (for JPEG)
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')

            optimized.append(frame)

        self.frames = optimized

    def add_watermark(
        self,
        text: str,
        position: Tuple[int, int] = None,
        opacity: int = 128
    ) -> None:
        """
        Add watermark to all frames.

        Args:
            text: Watermark text
            position: (x, y) position, defaults to bottom-right
            opacity: Opacity (0-255)
        """
        from PIL import ImageDraw, ImageFont

        for i, frame in enumerate(self.frames):
            # Create overlay
            overlay = Image.new('RGBA', frame.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            # Try to use a nice font, fallback to default
            try:
                font = ImageFont.truetype("Arial.ttf", 24)
            except:
                font = ImageFont.load_default()

            # Calculate position
            if position is None:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                position = (
                    frame.width - text_width - 20,
                    frame.height - text_height - 20
                )

            # Draw text
            draw.text(position, text, fill=(255, 255, 255, opacity), font=font)

            # Composite
            frame_rgba = frame.convert('RGBA')
            watermarked = Image.alpha_composite(frame_rgba, overlay)
            self.frames[i] = watermarked.convert('RGB')

    def get_frame_count(self) -> int:
        """Get number of frames generated."""
        return len(self.frames)

    def clear_frames(self) -> None:
        """Clear all frames from memory."""
        self.frames = []

    def estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of frames in bytes.

        Returns:
            Approximate memory usage in bytes
        """
        if not self.frames:
            return 0

        # Estimate bytes per pixel (RGB = 3 bytes)
        bytes_per_pixel = 3
        pixels_per_frame = self.width * self.height
        bytes_per_frame = pixels_per_frame * bytes_per_pixel

        return bytes_per_frame * len(self.frames)

    def estimate_file_size(self, format: str = 'png', quality: int = 85) -> int:
        """
        Estimate output file size.

        Args:
            format: Output format (png, jpg, gif)
            quality: Quality for lossy formats

        Returns:
            Approximate file size in bytes
        """
        if not self.frames:
            return 0

        # Sample first frame to estimate compression
        sample_frame = self.frames[0]
        buffer = io.BytesIO()

        if format.lower() == 'png':
            sample_frame.save(buffer, format='PNG', optimize=True)
        elif format.lower() in ['jpg', 'jpeg']:
            sample_frame.save(buffer, format='JPEG', quality=quality)
        elif format.lower() == 'gif':
            sample_frame.save(buffer, format='GIF', optimize=True)

        bytes_per_frame = buffer.tell()
        return bytes_per_frame * len(self.frames)


# Utility functions

def batch_generate_frames(
    charts: List,
    output_dir: str,
    width: int = 1920,
    height: int = 1080,
    prefix: str = "chart"
) -> List[str]:
    """
    Generate frames from multiple charts.

    Args:
        charts: List of VizForge chart objects
        output_dir: Output directory
        width: Frame width
        height: Frame height
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for i, chart in enumerate(charts):
        generator = FrameGenerator(chart, width, height)
        img_bytes = chart.fig.to_image(format="png", width=width, height=height)
        img = Image.open(io.BytesIO(img_bytes))

        filename = f"{prefix}_{i:05d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        paths.append(filepath)

    return paths
