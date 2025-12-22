"""
Animation engine for smooth chart transitions.

Provides easing functions and transition effects for video exports.
"""

from enum import Enum
from typing import List, Callable
import numpy as np


class AnimationType(Enum):
    """Animation transition types."""
    SMOOTH = "smooth"
    FADE = "fade"
    INSTANT = "instant"
    ELASTIC = "elastic"
    BOUNCE = "bounce"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"


class EasingFunction:
    """Mathematical easing functions for animations."""

    @staticmethod
    def linear(t: float) -> float:
        """Linear interpolation (no easing)."""
        return t

    @staticmethod
    def ease_in_quad(t: float) -> float:
        """Quadratic ease-in."""
        return t * t

    @staticmethod
    def ease_out_quad(t: float) -> float:
        """Quadratic ease-out."""
        return t * (2 - t)

    @staticmethod
    def ease_in_out_quad(t: float) -> float:
        """Quadratic ease-in-out."""
        return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t

    @staticmethod
    def ease_in_cubic(t: float) -> float:
        """Cubic ease-in."""
        return t * t * t

    @staticmethod
    def ease_out_cubic(t: float) -> float:
        """Cubic ease-out."""
        return (--t) * t * t + 1

    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        """Cubic ease-in-out."""
        return 4 * t * t * t if t < 0.5 else (t - 1) * (2 * t - 2) * (2 * t - 2) + 1

    @staticmethod
    def elastic(t: float) -> float:
        """Elastic easing (spring-like)."""
        if t == 0 or t == 1:
            return t
        p = 0.3
        s = p / 4
        return pow(2, -10 * t) * np.sin((t - s) * (2 * np.pi) / p) + 1

    @staticmethod
    def bounce(t: float) -> float:
        """Bounce easing."""
        if t < (1 / 2.75):
            return 7.5625 * t * t
        elif t < (2 / 2.75):
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < (2.5 / 2.75):
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625 / 2.75
            return 7.5625 * t * t + 0.984375


class AnimationEngine:
    """Animation engine for chart transitions."""

    EASING_MAP = {
        AnimationType.SMOOTH: EasingFunction.ease_in_out_cubic,
        AnimationType.FADE: EasingFunction.ease_out_quad,
        AnimationType.INSTANT: EasingFunction.linear,
        AnimationType.ELASTIC: EasingFunction.elastic,
        AnimationType.BOUNCE: EasingFunction.bounce,
        AnimationType.EASE_IN: EasingFunction.ease_in_cubic,
        AnimationType.EASE_OUT: EasingFunction.ease_out_cubic,
        AnimationType.EASE_IN_OUT: EasingFunction.ease_in_out_cubic,
    }

    @classmethod
    def get_easing_function(cls, animation_type: AnimationType) -> Callable:
        """Get easing function for animation type."""
        if isinstance(animation_type, str):
            animation_type = AnimationType(animation_type)
        return cls.EASING_MAP.get(animation_type, EasingFunction.ease_in_out_cubic)

    @classmethod
    def interpolate_values(
        cls,
        start_value: float,
        end_value: float,
        steps: int,
        animation_type: AnimationType = AnimationType.SMOOTH
    ) -> List[float]:
        """
        Interpolate between two values with easing.

        Args:
            start_value: Starting value
            end_value: Ending value
            steps: Number of interpolation steps
            animation_type: Type of animation/easing

        Returns:
            List of interpolated values
        """
        easing_func = cls.get_easing_function(animation_type)

        values = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 1.0
            eased_t = easing_func(t)
            value = start_value + (end_value - start_value) * eased_t
            values.append(value)

        return values

    @classmethod
    def generate_transition_frames(
        cls,
        start_data: np.ndarray,
        end_data: np.ndarray,
        n_frames: int,
        animation_type: AnimationType = AnimationType.SMOOTH
    ) -> List[np.ndarray]:
        """
        Generate transition frames between two data states.

        Args:
            start_data: Starting data array
            end_data: Ending data array
            n_frames: Number of frames to generate
            animation_type: Type of animation

        Returns:
            List of interpolated data arrays
        """
        easing_func = cls.get_easing_function(animation_type)

        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1) if n_frames > 1 else 1.0
            eased_t = easing_func(t)

            # Interpolate each element
            frame_data = start_data + (end_data - start_data) * eased_t
            frames.append(frame_data)

        return frames

    @classmethod
    def apply_fade_effect(
        cls,
        opacity_start: float,
        opacity_end: float,
        n_frames: int
    ) -> List[float]:
        """
        Generate opacity values for fade effect.

        Args:
            opacity_start: Starting opacity (0.0 to 1.0)
            opacity_end: Ending opacity (0.0 to 1.0)
            n_frames: Number of frames

        Returns:
            List of opacity values
        """
        return cls.interpolate_values(
            opacity_start,
            opacity_end,
            n_frames,
            AnimationType.FADE
        )

    @staticmethod
    def calculate_fps_from_duration(n_frames: int, duration_seconds: float) -> int:
        """Calculate FPS from number of frames and duration."""
        return max(1, int(n_frames / duration_seconds))

    @staticmethod
    def calculate_n_frames(fps: int, duration_seconds: float) -> int:
        """Calculate number of frames from FPS and duration."""
        return max(1, int(fps * duration_seconds))


class TransitionConfig:
    """Configuration for animation transitions."""

    def __init__(
        self,
        animation_type: AnimationType = AnimationType.SMOOTH,
        duration_ms: int = 500,
        fps: int = 30,
        interpolation_frames: int = None
    ):
        """
        Initialize transition configuration.

        Args:
            animation_type: Type of animation/easing
            duration_ms: Duration in milliseconds
            fps: Frames per second
            interpolation_frames: Number of frames to interpolate (auto-calculated if None)
        """
        self.animation_type = animation_type
        self.duration_ms = duration_ms
        self.fps = fps

        if interpolation_frames is None:
            self.interpolation_frames = int((duration_ms / 1000) * fps)
        else:
            self.interpolation_frames = interpolation_frames

    def get_easing_function(self) -> Callable:
        """Get the easing function for this configuration."""
        return AnimationEngine.get_easing_function(self.animation_type)


# Convenience functions

def smooth_transition(start: float, end: float, steps: int) -> List[float]:
    """Create smooth transition between two values."""
    return AnimationEngine.interpolate_values(start, end, steps, AnimationType.SMOOTH)


def elastic_transition(start: float, end: float, steps: int) -> List[float]:
    """Create elastic (spring-like) transition."""
    return AnimationEngine.interpolate_values(start, end, steps, AnimationType.ELASTIC)


def bounce_transition(start: float, end: float, steps: int) -> List[float]:
    """Create bounce transition."""
    return AnimationEngine.interpolate_values(start, end, steps, AnimationType.BOUNCE)


def fade_in(n_frames: int) -> List[float]:
    """Generate fade-in opacity values."""
    return AnimationEngine.apply_fade_effect(0.0, 1.0, n_frames)


def fade_out(n_frames: int) -> List[float]:
    """Generate fade-out opacity values."""
    return AnimationEngine.apply_fade_effect(1.0, 0.0, n_frames)
