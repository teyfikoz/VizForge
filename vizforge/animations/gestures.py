"""
VizForge Gestures

Touch and gesture support for mobile-friendly visualizations.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import plotly.graph_objects as go


class GestureType(Enum):
    """Types of gestures."""
    TAP = "tap"               # Single tap
    DOUBLE_TAP = "doubletap"  # Double tap
    LONG_PRESS = "longpress"  # Long press
    SWIPE = "swipe"           # Swipe gesture
    PINCH = "pinch"           # Pinch to zoom
    ROTATE = "rotate"         # Rotation gesture
    PAN = "pan"               # Pan/drag gesture


@dataclass
class TouchEvent:
    """
    Touch event data.

    Attributes:
        gesture_type: Type of gesture
        x: X coordinate
        y: Y coordinate
        timestamp: Event timestamp
        delta_x: X movement delta
        delta_y: Y movement delta
        scale: Pinch scale factor
        rotation: Rotation angle (degrees)
        velocity: Gesture velocity
    """
    gesture_type: GestureType
    x: float
    y: float
    timestamp: float
    delta_x: float = 0.0
    delta_y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    velocity: float = 0.0


@dataclass
class GestureConfig:
    """
    Configuration for gesture handling.

    Attributes:
        enabled_gestures: List of enabled gestures
        tap_threshold: Time threshold for tap (ms)
        double_tap_threshold: Time threshold for double tap (ms)
        long_press_threshold: Time threshold for long press (ms)
        swipe_threshold: Distance threshold for swipe (px)
        pinch_threshold: Scale threshold for pinch
        rotation_threshold: Angle threshold for rotation (degrees)
    """
    enabled_gestures: List[GestureType] = field(default_factory=lambda: [
        GestureType.TAP,
        GestureType.DOUBLE_TAP,
        GestureType.PINCH,
        GestureType.PAN
    ])
    tap_threshold: int = 200
    double_tap_threshold: int = 300
    long_press_threshold: int = 500
    swipe_threshold: int = 50
    pinch_threshold: float = 0.1
    rotation_threshold: float = 5.0


class GestureHandler:
    """
    Handle touch and gesture events.

    Provides mobile-friendly interaction patterns.

    Example:
        >>> handler = GestureHandler()
        >>> handler.on_tap(lambda event: print(f"Tapped at {event.x}, {event.y}"))
        >>> handler.on_pinch(lambda event: print(f"Pinch scale: {event.scale}"))
    """

    def __init__(self, config: Optional[GestureConfig] = None):
        """
        Initialize gesture handler.

        Args:
            config: Gesture configuration
        """
        self.config = config or GestureConfig()
        self._tap_handlers: List[Callable[[TouchEvent], None]] = []
        self._double_tap_handlers: List[Callable[[TouchEvent], None]] = []
        self._long_press_handlers: List[Callable[[TouchEvent], None]] = []
        self._swipe_handlers: List[Callable[[TouchEvent], None]] = []
        self._pinch_handlers: List[Callable[[TouchEvent], None]] = []
        self._rotate_handlers: List[Callable[[TouchEvent], None]] = []
        self._pan_handlers: List[Callable[[TouchEvent], None]] = []

    def on_tap(self, handler: Callable[[TouchEvent], None]) -> 'GestureHandler':
        """
        Register tap handler.

        Args:
            handler: Callback function(event)

        Returns:
            Self for method chaining

        Example:
            >>> handler.on_tap(lambda e: print(f"Tap at ({e.x}, {e.y})"))
        """
        self._tap_handlers.append(handler)
        return self

    def on_double_tap(self, handler: Callable[[TouchEvent], None]) -> 'GestureHandler':
        """
        Register double tap handler.

        Args:
            handler: Callback function(event)

        Returns:
            Self for method chaining
        """
        self._double_tap_handlers.append(handler)
        return self

    def on_long_press(self, handler: Callable[[TouchEvent], None]) -> 'GestureHandler':
        """
        Register long press handler.

        Args:
            handler: Callback function(event)

        Returns:
            Self for method chaining
        """
        self._long_press_handlers.append(handler)
        return self

    def on_swipe(self, handler: Callable[[TouchEvent], None]) -> 'GestureHandler':
        """
        Register swipe handler.

        Args:
            handler: Callback function(event)

        Returns:
            Self for method chaining

        Example:
            >>> handler.on_swipe(lambda e: print(f"Swiped {e.delta_x}px horizontally"))
        """
        self._swipe_handlers.append(handler)
        return self

    def on_pinch(self, handler: Callable[[TouchEvent], None]) -> 'GestureHandler':
        """
        Register pinch handler.

        Args:
            handler: Callback function(event)

        Returns:
            Self for method chaining

        Example:
            >>> handler.on_pinch(lambda e: zoom_chart(e.scale))
        """
        self._pinch_handlers.append(handler)
        return self

    def on_rotate(self, handler: Callable[[TouchEvent], None]) -> 'GestureHandler':
        """
        Register rotation handler.

        Args:
            handler: Callback function(event)

        Returns:
            Self for method chaining
        """
        self._rotate_handlers.append(handler)
        return self

    def on_pan(self, handler: Callable[[TouchEvent], None]) -> 'GestureHandler':
        """
        Register pan/drag handler.

        Args:
            handler: Callback function(event)

        Returns:
            Self for method chaining

        Example:
            >>> handler.on_pan(lambda e: pan_chart(e.delta_x, e.delta_y))
        """
        self._pan_handlers.append(handler)
        return self

    def handle_event(self, event: TouchEvent):
        """
        Dispatch touch event to registered handlers.

        Args:
            event: TouchEvent instance
        """
        handlers_map = {
            GestureType.TAP: self._tap_handlers,
            GestureType.DOUBLE_TAP: self._double_tap_handlers,
            GestureType.LONG_PRESS: self._long_press_handlers,
            GestureType.SWIPE: self._swipe_handlers,
            GestureType.PINCH: self._pinch_handlers,
            GestureType.ROTATE: self._rotate_handlers,
            GestureType.PAN: self._pan_handlers,
        }

        handlers = handlers_map.get(event.gesture_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in gesture handler: {e}")


# ==================== Helper Functions ====================

def setup_gestures(
    fig: go.Figure,
    enabled_gestures: Optional[List[GestureType]] = None
) -> go.Figure:
    """
    Enable gesture support on figure.

    Args:
        fig: Plotly figure
        enabled_gestures: List of enabled gestures (None = all)

    Returns:
        Figure with gesture support

    Example:
        >>> fig = setup_gestures(fig, [GestureType.TAP, GestureType.PINCH])
    """
    if enabled_gestures is None:
        enabled_gestures = [
            GestureType.TAP,
            GestureType.DOUBLE_TAP,
            GestureType.PINCH,
            GestureType.PAN
        ]

    # Configure drag modes
    config = {
        'scrollZoom': GestureType.PINCH in enabled_gestures,
        'doubleClick': 'reset' if GestureType.DOUBLE_TAP in enabled_gestures else False,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'responsive': True,
    }

    # Enable touch interactions
    fig.update_layout(
        dragmode='pan' if GestureType.PAN in enabled_gestures else False,
        hovermode='closest',
        clickmode='event+select'
    )

    # Store config in figure
    if not hasattr(fig, '_gesture_config'):
        fig._gesture_config = config

    return fig


def enable_mobile_gestures(
    fig: go.Figure,
    pinch_to_zoom: bool = True,
    tap_to_select: bool = True,
    double_tap_to_reset: bool = True,
    pan_to_navigate: bool = True
) -> go.Figure:
    """
    Quick setup for mobile-friendly gestures.

    Args:
        fig: Plotly figure
        pinch_to_zoom: Enable pinch to zoom
        tap_to_select: Enable tap to select points
        double_tap_to_reset: Enable double tap to reset view
        pan_to_navigate: Enable pan to navigate

    Returns:
        Figure with mobile gestures enabled

    Example:
        >>> fig = enable_mobile_gestures(fig)
    """
    enabled = []

    if tap_to_select:
        enabled.append(GestureType.TAP)

    if double_tap_to_reset:
        enabled.append(GestureType.DOUBLE_TAP)

    if pinch_to_zoom:
        enabled.append(GestureType.PINCH)

    if pan_to_navigate:
        enabled.append(GestureType.PAN)

    return setup_gestures(fig, enabled)


def create_touch_friendly_controls(
    fig: go.Figure,
    button_size: int = 40,
    button_color: str = '#4CAF50',
    position: str = 'top-right'
) -> go.Figure:
    """
    Add touch-friendly control buttons.

    Args:
        fig: Plotly figure
        button_size: Button size in pixels
        button_color: Button background color
        position: Button position ('top-right', 'top-left', 'bottom-right', 'bottom-left')

    Returns:
        Figure with touch controls

    Example:
        >>> fig = create_touch_friendly_controls(fig, button_size=50)
    """
    # Position mapping
    positions = {
        'top-right': {'x': 1.0, 'xanchor': 'right', 'y': 1.0, 'yanchor': 'top'},
        'top-left': {'x': 0.0, 'xanchor': 'left', 'y': 1.0, 'yanchor': 'top'},
        'bottom-right': {'x': 1.0, 'xanchor': 'right', 'y': 0.0, 'yanchor': 'bottom'},
        'bottom-left': {'x': 0.0, 'xanchor': 'left', 'y': 0.0, 'yanchor': 'bottom'},
    }

    pos_config = positions.get(position, positions['top-right'])

    # Update modebar
    fig.update_layout(
        modebar={
            'bgcolor': button_color,
            'color': 'white',
            'activecolor': '#2196F3',
            'orientation': 'v',
            **pos_config
        }
    )

    return fig


def optimize_for_touch(
    fig: go.Figure,
    min_touch_target: int = 44,
    increase_marker_size: bool = True,
    increase_line_width: bool = True,
    add_hover_labels: bool = True
) -> go.Figure:
    """
    Optimize chart for touch interactions.

    Follows mobile accessibility guidelines (44px minimum touch target).

    Args:
        fig: Plotly figure
        min_touch_target: Minimum touch target size (px)
        increase_marker_size: Increase marker sizes
        increase_line_width: Increase line widths
        add_hover_labels: Add hover labels

    Returns:
        Optimized figure

    Example:
        >>> fig = optimize_for_touch(fig, min_touch_target=48)
    """
    # Increase marker sizes
    if increase_marker_size:
        for trace in fig.data:
            if hasattr(trace, 'marker'):
                current_size = trace.marker.size if hasattr(trace.marker, 'size') else 6
                new_size = max(current_size, min_touch_target / 4)  # ~11px for 44px target
                trace.marker.update(size=new_size)

    # Increase line widths
    if increase_line_width:
        for trace in fig.data:
            if hasattr(trace, 'line'):
                current_width = trace.line.width if hasattr(trace.line, 'width') else 2
                new_width = max(current_width, 3)
                trace.line.update(width=new_width)

    # Add hover labels
    if add_hover_labels:
        fig.update_layout(
            hovermode='closest',
            hoverlabel=dict(
                bgcolor='white',
                font_size=14,
                font_family='Arial',
                bordercolor='black'
            )
        )

    return fig
