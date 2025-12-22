"""
Video export functionality for charts.

Supports MP4, WebM, and GIF formats.
"""

import os
import tempfile
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, List
import pandas as pd


class VideoFormat(Enum):
    """Supported video formats."""
    MP4 = "mp4"
    WEBM = "webm"
    GIF = "gif"
    FRAMES = "frames"  # Export individual frames


@dataclass
class VideoConfig:
    """Configuration for video export."""
    format: VideoFormat = VideoFormat.MP4
    fps: int = 30
    duration: float = 5.0
    width: int = 1920
    height: int = 1080
    quality: str = "high"  # low, medium, high
    loop: bool = True
    optimize: bool = True


class VideoExporter:
    """Export charts as animated videos."""

    def __init__(self, chart, config: VideoConfig = None):
        """
        Initialize video exporter.

        Args:
            chart: VizForge chart object
            config: Video configuration
        """
        self.chart = chart
        self.config = config or VideoConfig()
        self.frames = []

    def export(
        self,
        output_path: str,
        data_frames: List[pd.DataFrame] = None,
        transition: str = "smooth",
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Export chart as video.

        Args:
            output_path: Path to save video
            data_frames: List of DataFrames for animation frames
            transition: Transition type ('smooth', 'fade', 'instant')
            progress_callback: Optional callback for progress updates

        Returns:
            Path to exported video

        Example:
            >>> chart = vz.line(df, x='date', y='sales')
            >>> exporter = VideoExporter(chart)
            >>> exporter.export('sales_animation.mp4', data_frames=[df1, df2, df3])
        """
        if self.config.format == VideoFormat.GIF:
            return self._export_gif(output_path, data_frames, transition, progress_callback)
        elif self.config.format == VideoFormat.MP4:
            return self._export_mp4(output_path, data_frames, transition, progress_callback)
        elif self.config.format == VideoFormat.WEBM:
            return self._export_webm(output_path, data_frames, transition, progress_callback)
        elif self.config.format == VideoFormat.FRAMES:
            return self._export_frames(output_path, data_frames, progress_callback)
        else:
            raise ValueError(f"Unsupported format: {self.config.format}")

    def _export_gif(
        self,
        output_path: str,
        data_frames: List[pd.DataFrame],
        transition: str,
        progress_callback: Optional[Callable]
    ) -> str:
        """Export as GIF using PIL."""
        try:
            from PIL import Image
            import io
        except ImportError:
            raise ImportError(
                "GIF export requires 'Pillow'. Install with: pip install Pillow"
            )

        images = []
        total_frames = len(data_frames) if data_frames else 1

        for i, df in enumerate(data_frames or [self.chart.data]):
            if progress_callback:
                progress_callback(i / total_frames)

            # Update chart with new data
            updated_chart = self.chart.__class__(data=df, **self.chart._get_config())

            # Export frame as PNG
            img_bytes = updated_chart.fig.to_image(
                format="png",
                width=self.config.width,
                height=self.config.height
            )

            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)

        # Save as GIF
        if images:
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=int(1000 / self.config.fps),
                loop=0 if self.config.loop else 1,
                optimize=self.config.optimize
            )

        if progress_callback:
            progress_callback(1.0)

        return output_path

    def _export_mp4(
        self,
        output_path: str,
        data_frames: List[pd.DataFrame],
        transition: str,
        progress_callback: Optional[Callable]
    ) -> str:
        """Export as MP4 using ffmpeg."""
        import subprocess

        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "MP4 export requires 'ffmpeg'. "
                "Install from: https://ffmpeg.org/download.html"
            )

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_pattern = os.path.join(temp_dir, 'frame_%05d.png')
            total_frames = len(data_frames) if data_frames else 1

            # Generate frames
            for i, df in enumerate(data_frames or [self.chart.data]):
                if progress_callback:
                    progress_callback(i / (total_frames * 2))  # 50% for frame generation

                updated_chart = self.chart.__class__(data=df, **self.chart._get_config())

                frame_path = os.path.join(temp_dir, f'frame_{i:05d}.png')
                updated_chart.fig.write_image(
                    frame_path,
                    width=self.config.width,
                    height=self.config.height
                )

            # Encode video with ffmpeg
            quality_map = {
                'low': '28',
                'medium': '23',
                'high': '18'
            }
            crf = quality_map.get(self.config.quality, '23')

            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-framerate', str(self.config.fps),
                '-i', frame_pattern,
                '-c:v', 'libx264',
                '-crf', crf,
                '-pix_fmt', 'yuv420p',
                output_path
            ]

            subprocess.run(cmd, capture_output=True, check=True)

        if progress_callback:
            progress_callback(1.0)

        return output_path

    def _export_webm(
        self,
        output_path: str,
        data_frames: List[pd.DataFrame],
        transition: str,
        progress_callback: Optional[Callable]
    ) -> str:
        """Export as WebM using ffmpeg."""
        import subprocess

        # Similar to MP4 but with VP9 codec
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("WebM export requires 'ffmpeg'")

        with tempfile.TemporaryDirectory() as temp_dir:
            frame_pattern = os.path.join(temp_dir, 'frame_%05d.png')
            total_frames = len(data_frames) if data_frames else 1

            for i, df in enumerate(data_frames or [self.chart.data]):
                if progress_callback:
                    progress_callback(i / (total_frames * 2))

                updated_chart = self.chart.__class__(data=df, **self.chart._get_config())
                frame_path = os.path.join(temp_dir, f'frame_{i:05d}.png')
                updated_chart.fig.write_image(
                    frame_path,
                    width=self.config.width,
                    height=self.config.height
                )

            # Encode with VP9
            quality_map = {'low': '40', 'medium': '30', 'high': '20'}
            crf = quality_map.get(self.config.quality, '30')

            cmd = [
                'ffmpeg',
                '-y',
                '-framerate', str(self.config.fps),
                '-i', frame_pattern,
                '-c:v', 'libvpx-vp9',
                '-crf', crf,
                '-b:v', '0',
                output_path
            ]

            subprocess.run(cmd, capture_output=True, check=True)

        if progress_callback:
            progress_callback(1.0)

        return output_path

    def _export_frames(
        self,
        output_dir: str,
        data_frames: List[pd.DataFrame],
        progress_callback: Optional[Callable]
    ) -> str:
        """Export individual frames as PNG."""
        os.makedirs(output_dir, exist_ok=True)

        total_frames = len(data_frames) if data_frames else 1

        for i, df in enumerate(data_frames or [self.chart.data]):
            if progress_callback:
                progress_callback(i / total_frames)

            updated_chart = self.chart.__class__(data=df, **self.chart._get_config())

            frame_path = os.path.join(output_dir, f'frame_{i:05d}.png')
            updated_chart.fig.write_image(
                frame_path,
                width=self.config.width,
                height=self.config.height
            )

        if progress_callback:
            progress_callback(1.0)

        return output_dir


def export_video(
    chart,
    output_path: str,
    data_frames: List[pd.DataFrame] = None,
    format: str = 'mp4',
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    quality: str = 'high',
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Export chart as video (convenience function).

    Args:
        chart: VizForge chart
        output_path: Output file path
        data_frames: List of DataFrames for animation
        format: Video format ('mp4', 'webm', 'gif')
        fps: Frames per second
        width: Video width
        height: Video height
        quality: Quality level ('low', 'medium', 'high')
        progress_callback: Progress callback function

    Returns:
        Path to exported video

    Example:
        >>> chart = vz.line(df, x='date', y='sales')
        >>> vz.export_video(
        ...     chart,
        ...     'animation.mp4',
        ...     data_frames=[df_jan, df_feb, df_mar],
        ...     fps=30
        ... )
    """
    config = VideoConfig(
        format=VideoFormat(format),
        fps=fps,
        width=width,
        height=height,
        quality=quality
    )

    exporter = VideoExporter(chart, config)
    return exporter.export(output_path, data_frames, progress_callback=progress_callback)
