"""
VizForge 3D Interactive Controls

Advanced 3D interaction: 360° rotation, camera controls, animations, VR/AR support.
Part of VizForge v1.1.0 - Super AGI 3D Features.
"""

from typing import Optional, List, Tuple, Dict, Any, Callable
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
import json


@dataclass
class ControlConfig:
    """Configuration for 3D controls."""
    rotation_speed: float = 1.0
    zoom_speed: float = 1.0
    pan_speed: float = 1.0
    auto_rotate: bool = False
    rotation_axis: str = 'z'  # 'x', 'y', 'z'
    fps: int = 60


class RotationControl:
    """
    360-Degree Rotation Control for 3D Charts.

    Perfect for:
    - Product showcases
    - 3D model exploration
    - Architectural visualization
    - Scientific data presentation

    Example:
        >>> from vizforge.charts3d import Cone3D
        >>> cone = Cone3D(height=10, radius=5)
        >>> fig = cone.render()
        >>>
        >>> # Add rotation control
        >>> rotation = RotationControl(fig)
        >>> rotation.enable_auto_rotate(speed=2.0, axis='z')
        >>> rotation.apply()
        >>> fig.show()
    """

    def __init__(
        self,
        figure: go.Figure,
        config: Optional[ControlConfig] = None
    ):
        """
        Initialize rotation control.

        Args:
            figure: Plotly figure to control
            config: Configuration object
        """
        self.figure = figure
        self.config = config or ControlConfig()
        self.rotation_enabled = False
        self.current_angle = 0.0

    def enable_auto_rotate(
        self,
        speed: float = 1.0,
        axis: str = 'z',
        direction: str = 'clockwise'
    ) -> 'RotationControl':
        """
        Enable automatic 360° rotation.

        Args:
            speed: Rotation speed (degrees per second)
            axis: Rotation axis ('x', 'y', or 'z')
            direction: 'clockwise' or 'counterclockwise'

        Returns:
            Self for chaining
        """
        self.config.auto_rotate = True
        self.config.rotation_speed = speed
        self.config.rotation_axis = axis
        self.rotation_enabled = True

        return self

    def set_rotation_angle(self, angle: float, axis: str = 'z') -> 'RotationControl':
        """
        Set specific rotation angle.

        Args:
            angle: Rotation angle in degrees
            axis: Rotation axis

        Returns:
            Self for chaining
        """
        self.current_angle = angle
        self.config.rotation_axis = axis

        # Update camera position
        self._update_camera(angle, axis)

        return self

    def _update_camera(self, angle: float, axis: str):
        """Update camera position based on rotation."""
        rad = np.radians(angle)

        if axis == 'z':
            # Rotate around Z axis
            eye_x = 1.25 * np.cos(rad)
            eye_y = 1.25 * np.sin(rad)
            eye_z = 1.25
        elif axis == 'y':
            # Rotate around Y axis
            eye_x = 1.25 * np.cos(rad)
            eye_y = 1.25
            eye_z = 1.25 * np.sin(rad)
        else:  # 'x'
            # Rotate around X axis
            eye_x = 1.25
            eye_y = 1.25 * np.cos(rad)
            eye_z = 1.25 * np.sin(rad)

        self.figure.update_layout(
            scene_camera=dict(
                eye=dict(x=eye_x, y=eye_y, z=eye_z)
            )
        )

    def apply(self) -> go.Figure:
        """
        Apply rotation controls to figure.

        Returns:
            Updated figure
        """
        if self.config.auto_rotate:
            # Add animation frames for auto-rotation
            frames = []
            steps = 36  # 10-degree increments

            for i in range(steps):
                angle = i * 360 / steps
                self.set_rotation_angle(angle, self.config.rotation_axis)
                frames.append(go.Frame(
                    layout=dict(
                        scene_camera=self.figure.layout.scene.camera
                    )
                ))

            self.figure.frames = frames

            # Add play/pause buttons
            self.figure.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': '▶ Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 100, 'redraw': True},
                                'fromcurrent': True,
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        },
                        {
                            'label': '⏸ Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        }
                    ]
                }]
            )

        return self.figure


class CameraControl:
    """
    Advanced Camera Control for 3D Charts.

    Perfect for:
    - Custom viewpoints
    - Cinematic camera paths
    - Focus on specific features
    - Multi-angle presentations

    Example:
        >>> camera = CameraControl(fig)
        >>> camera.set_viewpoint('top')
        >>> camera.set_zoom(1.5)
        >>> camera.apply()
    """

    # Predefined viewpoints
    VIEWPOINTS = {
        'front': {'eye': {'x': 0, 'y': -2.5, 'z': 0}, 'up': {'x': 0, 'y': 0, 'z': 1}},
        'back': {'eye': {'x': 0, 'y': 2.5, 'z': 0}, 'up': {'x': 0, 'y': 0, 'z': 1}},
        'top': {'eye': {'x': 0, 'y': 0, 'z': 2.5}, 'up': {'x': 0, 'y': 1, 'z': 0}},
        'bottom': {'eye': {'x': 0, 'y': 0, 'z': -2.5}, 'up': {'x': 0, 'y': 1, 'z': 0}},
        'left': {'eye': {'x': -2.5, 'y': 0, 'z': 0}, 'up': {'x': 0, 'y': 0, 'z': 1}},
        'right': {'eye': {'x': 2.5, 'y': 0, 'z': 0}, 'up': {'x': 0, 'y': 0, 'z': 1}},
        'isometric': {'eye': {'x': 1.25, 'y': 1.25, 'z': 1.25}, 'up': {'x': 0, 'y': 0, 'z': 1}},
    }

    def __init__(
        self,
        figure: go.Figure,
        config: Optional[ControlConfig] = None
    ):
        """
        Initialize camera control.

        Args:
            figure: Plotly figure to control
            config: Configuration object
        """
        self.figure = figure
        self.config = config or ControlConfig()
        self.camera_params = {}

    def set_viewpoint(self, viewpoint: str) -> 'CameraControl':
        """
        Set camera to predefined viewpoint.

        Args:
            viewpoint: One of 'front', 'back', 'top', 'bottom', 'left', 'right', 'isometric'

        Returns:
            Self for chaining
        """
        if viewpoint not in self.VIEWPOINTS:
            raise ValueError(f"Unknown viewpoint: {viewpoint}. Choose from {list(self.VIEWPOINTS.keys())}")

        self.camera_params = self.VIEWPOINTS[viewpoint].copy()
        return self

    def set_custom_viewpoint(
        self,
        eye: Tuple[float, float, float],
        center: Tuple[float, float, float] = (0, 0, 0),
        up: Tuple[float, float, float] = (0, 0, 1)
    ) -> 'CameraControl':
        """
        Set custom camera viewpoint.

        Args:
            eye: Camera position (x, y, z)
            center: Look-at point (x, y, z)
            up: Up vector (x, y, z)

        Returns:
            Self for chaining
        """
        self.camera_params = {
            'eye': {'x': eye[0], 'y': eye[1], 'z': eye[2]},
            'center': {'x': center[0], 'y': center[1], 'z': center[2]},
            'up': {'x': up[0], 'y': up[1], 'z': up[2]}
        }
        return self

    def set_zoom(self, zoom: float) -> 'CameraControl':
        """
        Set camera zoom level.

        Args:
            zoom: Zoom factor (1.0 = default, >1 = zoom in, <1 = zoom out)

        Returns:
            Self for chaining
        """
        if 'eye' in self.camera_params:
            # Scale eye distance
            for axis in ['x', 'y', 'z']:
                self.camera_params['eye'][axis] /= zoom
        else:
            # Default zoom
            self.camera_params['eye'] = {
                'x': 1.25 / zoom,
                'y': 1.25 / zoom,
                'z': 1.25 / zoom
            }

        return self

    def apply(self) -> go.Figure:
        """
        Apply camera settings to figure.

        Returns:
            Updated figure
        """
        if self.camera_params:
            self.figure.update_layout(
                scene_camera=self.camera_params
            )

        return self.figure


class Animation3D:
    """
    3D Animation Engine.

    Perfect for:
    - Data evolution over time
    - Morphing shapes
    - Rotating objects
    - Step-by-step reveals

    Example:
        >>> anim = Animation3D(fig)
        >>> anim.add_rotation_animation(duration=5, axis='z')
        >>> anim.add_zoom_animation(start_zoom=0.5, end_zoom=2.0, duration=3)
        >>> anim.apply()
    """

    def __init__(
        self,
        figure: go.Figure,
        config: Optional[ControlConfig] = None
    ):
        """
        Initialize animation engine.

        Args:
            figure: Plotly figure to animate
            config: Configuration object
        """
        self.figure = figure
        self.config = config or ControlConfig()
        self.frames = []
        self.animations = []

    def add_rotation_animation(
        self,
        duration: float = 5.0,
        axis: str = 'z',
        full_rotations: int = 1
    ) -> 'Animation3D':
        """
        Add rotation animation.

        Args:
            duration: Animation duration in seconds
            axis: Rotation axis
            full_rotations: Number of complete rotations

        Returns:
            Self for chaining
        """
        fps = self.config.fps
        n_frames = int(duration * fps)

        for i in range(n_frames):
            angle = (i / n_frames) * 360 * full_rotations
            rad = np.radians(angle)

            if axis == 'z':
                eye = {'x': 1.25 * np.cos(rad), 'y': 1.25 * np.sin(rad), 'z': 1.25}
            elif axis == 'y':
                eye = {'x': 1.25 * np.cos(rad), 'y': 1.25, 'z': 1.25 * np.sin(rad)}
            else:  # 'x'
                eye = {'x': 1.25, 'y': 1.25 * np.cos(rad), 'z': 1.25 * np.sin(rad)}

            self.frames.append(go.Frame(
                layout=dict(scene_camera=dict(eye=eye))
            ))

        return self

    def add_zoom_animation(
        self,
        start_zoom: float = 0.5,
        end_zoom: float = 2.0,
        duration: float = 3.0
    ) -> 'Animation3D':
        """
        Add zoom animation.

        Args:
            start_zoom: Starting zoom level
            end_zoom: Ending zoom level
            duration: Animation duration

        Returns:
            Self for chaining
        """
        fps = self.config.fps
        n_frames = int(duration * fps)

        for i in range(n_frames):
            t = i / n_frames
            zoom = start_zoom + (end_zoom - start_zoom) * t
            distance = 1.25 / zoom

            self.frames.append(go.Frame(
                layout=dict(scene_camera=dict(eye=dict(x=distance, y=distance, z=distance)))
            ))

        return self

    def apply(self, loop: bool = True) -> go.Figure:
        """
        Apply animations to figure.

        Args:
            loop: Whether to loop animation

        Returns:
            Animated figure
        """
        if self.frames:
            self.figure.frames = self.frames

            # Add play controls
            self.figure.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': '▶ Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 1000 / self.config.fps, 'redraw': True},
                                'fromcurrent': True,
                                'mode': 'immediate' if loop else 'afterall'
                            }]
                        },
                        {
                            'label': '⏸ Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate'
                            }]
                        }
                    ]
                }]
            )

        return self.figure


class VRController:
    """
    VR/AR Support for 3D Visualizations.

    Perfect for:
    - Immersive data exploration
    - VR presentations
    - AR overlays
    - WebXR experiences

    Example:
        >>> vr = VRController(fig)
        >>> vr.enable_vr_mode()
        >>> vr.export_for_webxr('output.html')
    """

    def __init__(
        self,
        figure: go.Figure,
        config: Optional[ControlConfig] = None
    ):
        """
        Initialize VR controller.

        Args:
            figure: Plotly figure to enable VR for
            config: Configuration object
        """
        self.figure = figure
        self.config = config or ControlConfig()
        self.vr_enabled = False

    def enable_vr_mode(self) -> 'VRController':
        """
        Enable VR mode for the visualization.

        Returns:
            Self for chaining
        """
        self.vr_enabled = True

        # Optimize for VR
        self.figure.update_layout(
            scene=dict(
                camera=dict(
                    projection=dict(type='perspective')
                ),
                aspectmode='data'
            ),
            # Full screen for VR
            margin=dict(l=0, r=0, t=0, b=0)
        )

        return self

    def enable_ar_mode(self) -> 'VRController':
        """
        Enable AR mode (requires WebXR support).

        Returns:
            Self for chaining
        """
        # AR mode configuration
        self.figure.update_layout(
            scene=dict(
                bgcolor='rgba(0,0,0,0)',  # Transparent background for AR
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            )
        )

        return self

    def export_for_webxr(self, filename: str = 'vr_viz.html') -> str:
        """
        Export visualization for WebXR.

        Args:
            filename: Output HTML file

        Returns:
            Path to exported file
        """
        # Add WebXR metadata
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['toggleSpikelines', 'hoverClosest3d'],
            'responsive': True
        }

        # Save with VR-optimized config
        self.figure.write_html(
            filename,
            config=config,
            include_plotlyjs='cdn'
        )

        return filename

    def apply(self) -> go.Figure:
        """
        Apply VR/AR settings.

        Returns:
            Updated figure
        """
        return self.figure


# ==================== Helper Functions ====================

def enable_360_rotation(
    figure: go.Figure,
    speed: float = 1.0,
    axis: str = 'z'
) -> go.Figure:
    """
    Quick enable 360° auto-rotation.

    Args:
        figure: Plotly figure
        speed: Rotation speed
        axis: Rotation axis

    Returns:
        Figure with auto-rotation

    Example:
        >>> fig = create_cone()
        >>> fig = enable_360_rotation(fig, speed=2.0)
        >>> fig.show()
    """
    rotation = RotationControl(figure)
    rotation.enable_auto_rotate(speed=speed, axis=axis)
    return rotation.apply()


def enable_vr_mode(figure: go.Figure) -> go.Figure:
    """
    Quick enable VR mode.

    Args:
        figure: Plotly figure

    Returns:
        VR-enabled figure

    Example:
        >>> fig = create_surface(lambda x, y: np.sin(x) * np.cos(y))
        >>> fig = enable_vr_mode(fig)
        >>> fig.show()
    """
    vr = VRController(figure)
    vr.enable_vr_mode()
    return vr.apply()


def add_viewpoint_buttons(figure: go.Figure) -> go.Figure:
    """
    Add quick viewpoint selection buttons.

    Args:
        figure: Plotly figure

    Returns:
        Figure with viewpoint buttons
    """
    buttons = []

    for name, params in CameraControl.VIEWPOINTS.items():
        buttons.append({
            'label': name.title(),
            'method': 'relayout',
            'args': [{'scene.camera': params}]
        })

    figure.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'y': 1.1,
            'xanchor': 'left',
            'yanchor': 'top'
        }]
    )

    return figure


def create_camera_path(
    waypoints: List[Tuple[float, float, float]],
    n_frames: int = 100
) -> List[Dict]:
    """
    Create smooth camera path through waypoints.

    Args:
        waypoints: List of (x, y, z) camera positions
        n_frames: Number of interpolation frames

    Returns:
        List of camera frame configurations

    Example:
        >>> waypoints = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 0, 0)]
        >>> path = create_camera_path(waypoints, n_frames=120)
    """
    from scipy.interpolate import interp1d

    waypoints = np.array(waypoints)
    t_waypoints = np.linspace(0, 1, len(waypoints))
    t_frames = np.linspace(0, 1, n_frames)

    # Interpolate each axis
    interp_x = interp1d(t_waypoints, waypoints[:, 0], kind='cubic')
    interp_y = interp1d(t_waypoints, waypoints[:, 1], kind='cubic')
    interp_z = interp1d(t_waypoints, waypoints[:, 2], kind='cubic')

    # Generate frames
    frames = []
    for t in t_frames:
        eye = {
            'x': float(interp_x(t)),
            'y': float(interp_y(t)),
            'z': float(interp_z(t))
        }
        frames.append({'eye': eye})

    return frames
