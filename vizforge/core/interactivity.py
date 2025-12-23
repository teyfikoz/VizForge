"""
Enhanced Interactivity for VizForge

Advanced interaction patterns that make VizForge superior to Plotly:
- Touch and gesture support (pinch, swipe, rotate)
- 3D navigation (orbit, pan, zoom)
- Smart zoom (semantic zoom levels)
- Multi-touch gestures
- Haptic feedback integration

Plotly limitation: Basic mouse-only interactions.
VizForge innovation: Full touch, gesture, and 3D control.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class GestureType(Enum):
    """Supported gesture types."""
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH_IN = "pinch_in"
    PINCH_OUT = "pinch_out"
    ROTATE_CW = "rotate_cw"
    ROTATE_CCW = "rotate_ccw"
    TWO_FINGER_DRAG = "two_finger_drag"


@dataclass
class TouchPoint:
    """Represents a single touch point."""
    id: int
    x: float
    y: float
    timestamp: float
    pressure: float = 1.0


@dataclass
class GestureEvent:
    """Gesture event data."""
    gesture_type: GestureType
    start_points: List[TouchPoint]
    end_points: List[TouchPoint]
    duration: float
    velocity: Optional[Tuple[float, float]] = None
    scale: Optional[float] = None
    rotation: Optional[float] = None


class GestureRecognizer:
    """
    Multi-touch gesture recognition system.

    Recognizes common touch gestures and converts them to chart actions.
    """

    def __init__(self,
                 tap_threshold: float = 200,  # milliseconds
                 long_press_threshold: float = 500,
                 swipe_threshold: float = 50,  # pixels
                 pinch_threshold: float = 0.1):  # scale change
        """
        Initialize gesture recognizer.

        Args:
            tap_threshold: Max duration for tap gesture (ms)
            long_press_threshold: Min duration for long press (ms)
            swipe_threshold: Min distance for swipe (pixels)
            pinch_threshold: Min scale change for pinch
        """
        self.tap_threshold = tap_threshold / 1000.0  # Convert to seconds
        self.long_press_threshold = long_press_threshold / 1000.0
        self.swipe_threshold = swipe_threshold
        self.pinch_threshold = pinch_threshold

        self.active_touches: Dict[int, TouchPoint] = {}
        self.gesture_callbacks: Dict[GestureType, List[Callable]] = {}

    def on_touch_start(self, touch: TouchPoint):
        """Handle touch start event."""
        self.active_touches[touch.id] = touch

    def on_touch_move(self, touch: TouchPoint):
        """Handle touch move event."""
        if touch.id in self.active_touches:
            self.active_touches[touch.id] = touch

    def on_touch_end(self, touch: TouchPoint):
        """Handle touch end event and recognize gesture."""
        if touch.id not in self.active_touches:
            return

        start_touch = self.active_touches[touch.id]
        duration = touch.timestamp - start_touch.timestamp

        # Calculate movement
        dx = touch.x - start_touch.x
        dy = touch.y - start_touch.y
        distance = np.sqrt(dx**2 + dy**2)

        # Recognize gesture
        gesture = self._recognize_single_touch(start_touch, touch, duration, dx, dy, distance)

        if gesture:
            self._trigger_gesture(gesture)

        del self.active_touches[touch.id]

    def _recognize_single_touch(self,
                                start: TouchPoint,
                                end: TouchPoint,
                                duration: float,
                                dx: float,
                                dy: float,
                                distance: float) -> Optional[GestureEvent]:
        """Recognize single-touch gesture."""

        # Tap or Long Press
        if distance < 10:  # Minimal movement
            if duration < self.tap_threshold:
                return GestureEvent(
                    gesture_type=GestureType.TAP,
                    start_points=[start],
                    end_points=[end],
                    duration=duration
                )
            elif duration >= self.long_press_threshold:
                return GestureEvent(
                    gesture_type=GestureType.LONG_PRESS,
                    start_points=[start],
                    end_points=[end],
                    duration=duration
                )

        # Swipe
        if distance >= self.swipe_threshold:
            velocity = (dx / duration, dy / duration)

            # Determine swipe direction
            angle = np.arctan2(dy, dx)

            if -np.pi/4 <= angle < np.pi/4:
                gesture_type = GestureType.SWIPE_RIGHT
            elif np.pi/4 <= angle < 3*np.pi/4:
                gesture_type = GestureType.SWIPE_DOWN
            elif angle >= 3*np.pi/4 or angle < -3*np.pi/4:
                gesture_type = GestureType.SWIPE_LEFT
            else:
                gesture_type = GestureType.SWIPE_UP

            return GestureEvent(
                gesture_type=gesture_type,
                start_points=[start],
                end_points=[end],
                duration=duration,
                velocity=velocity
            )

        return None

    def recognize_multi_touch(self, touches: List[TouchPoint]) -> Optional[GestureEvent]:
        """Recognize multi-touch gestures (pinch, rotate)."""
        if len(touches) < 2:
            return None

        # Get first two touches
        t1, t2 = touches[0], touches[1]

        # Check if we have start positions
        if t1.id not in self.active_touches or t2.id not in self.active_touches:
            return None

        s1 = self.active_touches[t1.id]
        s2 = self.active_touches[t2.id]

        # Calculate initial and current distances
        start_distance = np.sqrt((s2.x - s1.x)**2 + (s2.y - s1.y)**2)
        current_distance = np.sqrt((t2.x - t1.x)**2 + (t2.y - t1.y)**2)

        # Pinch detection
        scale = current_distance / start_distance if start_distance > 0 else 1.0

        if abs(scale - 1.0) > self.pinch_threshold:
            gesture_type = GestureType.PINCH_OUT if scale > 1.0 else GestureType.PINCH_IN

            return GestureEvent(
                gesture_type=gesture_type,
                start_points=[s1, s2],
                end_points=[t1, t2],
                duration=t1.timestamp - s1.timestamp,
                scale=scale
            )

        # Rotation detection
        start_angle = np.arctan2(s2.y - s1.y, s2.x - s1.x)
        current_angle = np.arctan2(t2.y - t1.y, t2.x - t1.x)
        rotation = current_angle - start_angle

        if abs(rotation) > 0.1:  # ~6 degrees
            gesture_type = GestureType.ROTATE_CW if rotation > 0 else GestureType.ROTATE_CCW

            return GestureEvent(
                gesture_type=gesture_type,
                start_points=[s1, s2],
                end_points=[t1, t2],
                duration=t1.timestamp - s1.timestamp,
                rotation=rotation
            )

        return None

    def on_gesture(self, gesture_type: GestureType, callback: Callable):
        """Register callback for gesture type."""
        if gesture_type not in self.gesture_callbacks:
            self.gesture_callbacks[gesture_type] = []
        self.gesture_callbacks[gesture_type].append(callback)

    def _trigger_gesture(self, event: GestureEvent):
        """Trigger callbacks for recognized gesture."""
        if event.gesture_type in self.gesture_callbacks:
            for callback in self.gesture_callbacks[event.gesture_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Gesture callback error: {e}")


class Navigation3D:
    """
    Advanced 3D navigation system.

    Provides intuitive 3D chart navigation with orbit, pan, and zoom.
    """

    def __init__(self):
        """Initialize 3D navigation."""
        self.camera_position = np.array([0.0, 0.0, 10.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 1.0, 0.0])

        self.orbit_speed = 0.01
        self.pan_speed = 0.1
        self.zoom_speed = 0.1

        self.min_distance = 1.0
        self.max_distance = 100.0

    def orbit(self, delta_x: float, delta_y: float):
        """
        Orbit camera around target.

        Args:
            delta_x: Horizontal rotation (radians)
            delta_y: Vertical rotation (radians)
        """
        # Get vector from target to camera
        offset = self.camera_position - self.camera_target
        distance = np.linalg.norm(offset)

        # Convert to spherical coordinates
        theta = np.arctan2(offset[0], offset[2])
        phi = np.arccos(offset[1] / distance)

        # Apply rotation
        theta += delta_x * self.orbit_speed
        phi = np.clip(phi + delta_y * self.orbit_speed, 0.01, np.pi - 0.01)

        # Convert back to Cartesian
        self.camera_position = self.camera_target + distance * np.array([
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
            np.sin(phi) * np.cos(theta)
        ])

    def pan(self, delta_x: float, delta_y: float):
        """
        Pan camera parallel to view plane.

        Args:
            delta_x: Horizontal pan
            delta_y: Vertical pan
        """
        # Get camera right and up vectors
        forward = self.camera_target - self.camera_position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.camera_up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Apply pan
        offset = (right * delta_x + up * delta_y) * self.pan_speed

        self.camera_position += offset
        self.camera_target += offset

    def zoom(self, delta: float):
        """
        Zoom camera toward/away from target.

        Args:
            delta: Zoom amount (positive = zoom in)
        """
        direction = self.camera_target - self.camera_position
        distance = np.linalg.norm(direction)
        direction = direction / distance

        # Calculate new distance
        new_distance = distance - delta * self.zoom_speed * distance
        new_distance = np.clip(new_distance, self.min_distance, self.max_distance)

        # Update camera position
        self.camera_position = self.camera_target - direction * new_distance

    def reset(self):
        """Reset camera to default position."""
        self.camera_position = np.array([0.0, 0.0, 10.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 1.0, 0.0])

    def get_view_matrix(self) -> np.ndarray:
        """
        Get view matrix for rendering.

        Returns:
            4x4 view matrix
        """
        # Look-at matrix
        forward = self.camera_target - self.camera_position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.camera_up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        view_matrix = np.eye(4)
        view_matrix[0, :3] = right
        view_matrix[1, :3] = up
        view_matrix[2, :3] = -forward
        view_matrix[:3, 3] = -np.array([
            np.dot(right, self.camera_position),
            np.dot(up, self.camera_position),
            np.dot(forward, self.camera_position)
        ])

        return view_matrix


class SemanticZoom:
    """
    Semantic zoom that changes visualization detail level.

    Unlike simple geometric zoom, semantic zoom adapts the visualization
    based on zoom level (e.g., show/hide labels, aggregate data).
    """

    def __init__(self, levels: int = 5):
        """
        Initialize semantic zoom.

        Args:
            levels: Number of detail levels
        """
        self.levels = levels
        self.current_level = levels // 2
        self.level_callbacks: Dict[int, List[Callable]] = {}

    def zoom_in(self):
        """Increase detail level."""
        if self.current_level < self.levels - 1:
            self.current_level += 1
            self._trigger_level_change()

    def zoom_out(self):
        """Decrease detail level."""
        if self.current_level > 0:
            self.current_level -= 1
            self._trigger_level_change()

    def set_level(self, level: int):
        """
        Set detail level directly.

        Args:
            level: Detail level (0 to levels-1)
        """
        if 0 <= level < self.levels:
            self.current_level = level
            self._trigger_level_change()

    def on_level(self, level: int, callback: Callable):
        """
        Register callback for detail level.

        Args:
            level: Detail level
            callback: Function to call when level is reached
        """
        if level not in self.level_callbacks:
            self.level_callbacks[level] = []
        self.level_callbacks[level].append(callback)

    def _trigger_level_change(self):
        """Trigger callbacks for current level."""
        if self.current_level in self.level_callbacks:
            for callback in self.level_callbacks[self.current_level]:
                try:
                    callback(self.current_level)
                except Exception as e:
                    print(f"Level callback error: {e}")


class HapticFeedback:
    """
    Haptic feedback integration for mobile devices.

    Provides tactile feedback for interactions.
    """

    class Intensity(Enum):
        """Haptic intensity levels."""
        LIGHT = "light"
        MEDIUM = "medium"
        HEAVY = "heavy"

    class Pattern(Enum):
        """Haptic patterns."""
        SINGLE_TAP = "single_tap"
        DOUBLE_TAP = "double_tap"
        SUCCESS = "success"
        WARNING = "warning"
        ERROR = "error"

    @staticmethod
    def trigger(pattern: Pattern, intensity: Intensity = Intensity.MEDIUM):
        """
        Trigger haptic feedback.

        Args:
            pattern: Haptic pattern
            intensity: Vibration intensity
        """
        # In production, this would interface with device APIs
        # For now, just log
        print(f"Haptic: {pattern.value} ({intensity.value})")


# Helper functions for chart integration
def enable_gestures(chart,
                   enable_pinch_zoom: bool = True,
                   enable_swipe_pan: bool = True,
                   enable_rotate: bool = False):
    """
    Enable gesture controls for a chart.

    Args:
        chart: Chart object
        enable_pinch_zoom: Enable pinch to zoom
        enable_swipe_pan: Enable swipe to pan
        enable_rotate: Enable rotation gesture

    Example:
        chart = vz.scatter3d(df, x='x', y='y', z='z')
        enable_gestures(chart, enable_rotate=True)
    """
    recognizer = GestureRecognizer()

    if enable_pinch_zoom:
        def on_pinch(event: GestureEvent):
            if event.scale:
                # Apply zoom based on pinch scale
                current_range = chart.fig.layout.scene.camera.eye
                chart.fig.layout.scene.camera.eye.z *= (2.0 - event.scale)

        recognizer.on_gesture(GestureType.PINCH_IN, on_pinch)
        recognizer.on_gesture(GestureType.PINCH_OUT, on_pinch)

    if enable_swipe_pan:
        def on_swipe(event: GestureEvent):
            if event.velocity:
                # Apply pan based on swipe velocity
                dx, dy = event.velocity
                chart.fig.layout.scene.camera.center.x += dx * 0.01
                chart.fig.layout.scene.camera.center.y += dy * 0.01

        recognizer.on_gesture(GestureType.SWIPE_LEFT, on_swipe)
        recognizer.on_gesture(GestureType.SWIPE_RIGHT, on_swipe)
        recognizer.on_gesture(GestureType.SWIPE_UP, on_swipe)
        recognizer.on_gesture(GestureType.SWIPE_DOWN, on_swipe)

    if enable_rotate:
        def on_rotate(event: GestureEvent):
            if event.rotation:
                # Apply rotation
                # This would update camera rotation in production
                pass

        recognizer.on_gesture(GestureType.ROTATE_CW, on_rotate)
        recognizer.on_gesture(GestureType.ROTATE_CCW, on_rotate)

    chart._gesture_recognizer = recognizer
    return recognizer


def enable_3d_navigation(chart,
                        orbit_speed: float = 0.01,
                        pan_speed: float = 0.1,
                        zoom_speed: float = 0.1):
    """
    Enable advanced 3D navigation for 3D charts.

    Args:
        chart: 3D chart object
        orbit_speed: Orbit sensitivity
        pan_speed: Pan sensitivity
        zoom_speed: Zoom sensitivity

    Example:
        chart = vz.scatter3d(df, x='x', y='y', z='z')
        nav = enable_3d_navigation(chart)
    """
    nav = Navigation3D()
    nav.orbit_speed = orbit_speed
    nav.pan_speed = pan_speed
    nav.zoom_speed = zoom_speed

    chart._navigation_3d = nav
    return nav


def enable_semantic_zoom(chart, levels: int = 5):
    """
    Enable semantic zoom with detail level adaptation.

    Args:
        chart: Chart object
        levels: Number of detail levels

    Example:
        chart = vz.scatter(df, x='x', y='y')
        zoom = enable_semantic_zoom(chart, levels=3)

        @zoom.on_level(0)  # Lowest detail
        def show_aggregated():
            # Show aggregated view
            pass

        @zoom.on_level(2)  # Highest detail
        def show_all_points():
            # Show all data points
            pass
    """
    zoom = SemanticZoom(levels=levels)
    chart._semantic_zoom = zoom
    return zoom
