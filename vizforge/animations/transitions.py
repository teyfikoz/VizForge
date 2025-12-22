"""
VizForge Transitions

Smooth chart transitions and animation configurations.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import plotly.graph_objects as go
from .easing import get_easing_function, EasingFunction


class TransitionType(Enum):
    """Types of transitions."""
    SMOOTH = "smooth"           # Smooth transition (default)
    INSTANT = "instant"         # No transition
    FADE = "fade"              # Fade in/out
    SLIDE = "slide"            # Slide from direction
    ZOOM = "zoom"              # Zoom in/out
    ELASTIC = "elastic"        # Elastic bounce
    BOUNCE = "bounce"          # Bounce effect


@dataclass
class TransitionConfig:
    """
    Configuration for chart transitions.

    Attributes:
        transition_type: Type of transition
        duration: Duration in milliseconds
        easing: Easing function name
        ordering: Ordering of animated elements ('layout first', 'traces first', 'together')
        mode: Transition mode ('immediate', 'gradual', 'next')
        redraw: Whether to redraw chart
    """
    transition_type: TransitionType = TransitionType.SMOOTH
    duration: int = 500
    easing: str = 'ease-in-out'
    ordering: str = 'layout first'
    mode: str = 'immediate'
    redraw: bool = True

    # Advanced options
    frame_duration: Optional[int] = None
    frame_redraw: bool = False
    from_current: bool = False


class AnimationEngine:
    """
    Engine for managing chart animations.

    Handles transition application, frame generation, and animation playback.

    Example:
        >>> engine = AnimationEngine()
        >>> config = TransitionConfig(transition_type=TransitionType.ELASTIC, duration=800)
        >>> engine.apply_transition(fig, config)
    """

    @staticmethod
    def apply_transition(
        fig: go.Figure,
        config: TransitionConfig
    ) -> go.Figure:
        """
        Apply transition configuration to figure.

        Args:
            fig: Plotly figure
            config: Transition configuration

        Returns:
            Figure with transitions applied

        Example:
            >>> config = TransitionConfig(duration=1000, easing='ease-out-bounce')
            >>> fig = AnimationEngine.apply_transition(fig, config)
        """
        # Build Plotly transition dict
        transition = {
            'duration': config.duration,
            'easing': config.easing,
            'ordering': config.ordering,
            'redraw': config.redraw
        }

        # Apply to layout
        fig.update_layout(
            transition=transition
        )

        # Apply to traces if specified
        if config.mode == 'gradual':
            for trace in fig.data:
                trace.update(
                    transition=transition
                )

        return fig

    @staticmethod
    def create_frames(
        data_states: List[Dict[str, Any]],
        frame_duration: int = 500,
        transition_duration: int = 500,
        easing: str = 'ease-in-out'
    ) -> List[go.Frame]:
        """
        Create animation frames from data states.

        Args:
            data_states: List of data dictionaries for each frame
            frame_duration: Duration each frame displays
            transition_duration: Duration of transition between frames
            easing: Easing function name

        Returns:
            List of Plotly frames

        Example:
            >>> states = [
            ...     {'x': [1, 2, 3], 'y': [1, 2, 3]},
            ...     {'x': [1, 2, 3], 'y': [2, 3, 4]},
            ... ]
            >>> frames = AnimationEngine.create_frames(states)
        """
        frames = []

        for i, state in enumerate(data_states):
            frame = go.Frame(
                data=[go.Scatter(**state)],
                name=f'frame_{i}',
                layout=go.Layout(
                    transition={
                        'duration': transition_duration,
                        'easing': easing
                    }
                )
            )
            frames.append(frame)

        return frames

    @staticmethod
    def add_play_button(
        fig: go.Figure,
        frame_duration: int = 500,
        transition_duration: int = 500,
        x_position: float = 0.1,
        y_position: float = 0.0
    ) -> go.Figure:
        """
        Add play/pause buttons to animated chart.

        Args:
            fig: Plotly figure with frames
            frame_duration: Duration per frame
            transition_duration: Transition duration
            x_position: X position of buttons (0.0 to 1.0)
            y_position: Y position of buttons (0.0 to 1.0)

        Returns:
            Figure with play controls

        Example:
            >>> fig = go.Figure(data=[...], frames=[...])
            >>> fig = AnimationEngine.add_play_button(fig)
        """
        fig.update_layout(
            updatemenus=[
                {
                    'type': 'buttons',
                    'showactive': False,
                    'x': x_position,
                    'y': y_position,
                    'xanchor': 'left',
                    'yanchor': 'bottom',
                    'buttons': [
                        {
                            'label': '▶ Play',
                            'method': 'animate',
                            'args': [
                                None,
                                {
                                    'frame': {'duration': frame_duration, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {
                                        'duration': transition_duration,
                                        'easing': 'ease-in-out'
                                    },
                                    'mode': 'immediate'
                                }
                            ]
                        },
                        {
                            'label': '⏸ Pause',
                            'method': 'animate',
                            'args': [
                                [None],
                                {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }
                            ]
                        }
                    ]
                }
            ]
        )

        return fig

    @staticmethod
    def add_slider(
        fig: go.Figure,
        frame_names: Optional[List[str]] = None,
        slider_len: float = 0.9,
        x_position: float = 0.1,
        y_position: float = 0.0
    ) -> go.Figure:
        """
        Add timeline slider to animated chart.

        Args:
            fig: Plotly figure with frames
            frame_names: Custom frame names for slider
            slider_len: Slider length (0.0 to 1.0)
            x_position: X position of slider
            y_position: Y position of slider

        Returns:
            Figure with slider

        Example:
            >>> fig = AnimationEngine.add_slider(fig, frame_names=['2020', '2021', '2022'])
        """
        if frame_names is None:
            frame_names = [f'Frame {i}' for i in range(len(fig.frames))]

        steps = []
        for i, frame_name in enumerate(frame_names):
            step = {
                'args': [
                    [fig.frames[i].name],
                    {
                        'frame': {'duration': 500, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 500}
                    }
                ],
                'label': frame_name,
                'method': 'animate'
            }
            steps.append(step)

        sliders = [{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Frame: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 500, 'easing': 'ease-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': slider_len,
            'x': x_position,
            'y': y_position,
            'steps': steps
        }]

        fig.update_layout(sliders=sliders)

        return fig

    @staticmethod
    def smooth_update(
        fig: go.Figure,
        new_data: Dict[str, Any],
        duration: int = 500,
        easing: str = 'ease-in-out'
    ) -> go.Figure:
        """
        Smoothly update chart data with transition.

        Args:
            fig: Plotly figure
            new_data: New data to update
            duration: Transition duration
            easing: Easing function

        Returns:
            Updated figure

        Example:
            >>> new_data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
            >>> fig = AnimationEngine.smooth_update(fig, new_data, duration=1000)
        """
        # Create frame with new data
        frame = go.Frame(
            data=[go.Scatter(**new_data)],
            layout=go.Layout(
                transition={'duration': duration, 'easing': easing}
            )
        )

        # Add frame and trigger animation
        fig.frames = [frame]

        return fig


# ==================== Helper Functions ====================

def apply_transition(
    fig: go.Figure,
    transition: Union[str, TransitionType, TransitionConfig] = 'smooth',
    duration: int = 500,
    easing: str = 'ease-in-out'
) -> go.Figure:
    """
    Quick apply transition to figure.

    Args:
        fig: Plotly figure
        transition: Transition type or config
        duration: Duration in milliseconds
        easing: Easing function name

    Returns:
        Figure with transition applied

    Example:
        >>> fig = apply_transition(fig, 'elastic', duration=800)
        >>> fig = apply_transition(fig, TransitionType.BOUNCE, duration=1000)
    """
    # Convert string to TransitionType
    if isinstance(transition, str):
        try:
            transition = TransitionType(transition.lower())
        except ValueError:
            transition = TransitionType.SMOOTH

    # Create config if not provided
    if not isinstance(transition, TransitionConfig):
        config = TransitionConfig(
            transition_type=transition if isinstance(transition, TransitionType) else TransitionType.SMOOTH,
            duration=duration,
            easing=easing
        )
    else:
        config = transition

    # Apply via engine
    return AnimationEngine.apply_transition(fig, config)


def create_transition(
    transition_type: Union[str, TransitionType] = 'smooth',
    duration: int = 500,
    easing: str = 'ease-in-out',
    **kwargs
) -> TransitionConfig:
    """
    Create transition configuration.

    Args:
        transition_type: Type of transition
        duration: Duration in milliseconds
        easing: Easing function name
        **kwargs: Additional transition options

    Returns:
        TransitionConfig instance

    Example:
        >>> config = create_transition('elastic', duration=800, easing='ease-out-bounce')
        >>> fig = apply_transition(fig, config)
    """
    # Convert string to TransitionType
    if isinstance(transition_type, str):
        try:
            transition_type = TransitionType(transition_type.lower())
        except ValueError:
            transition_type = TransitionType.SMOOTH

    return TransitionConfig(
        transition_type=transition_type,
        duration=duration,
        easing=easing,
        **kwargs
    )
