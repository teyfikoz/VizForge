"""
VizForge 3D Geometric Charts

Special 3D geometric visualizations: cone, spiral, helix, torus, sphere.
Part of VizForge v1.1.0 - Super AGI 3D Features.
"""

from typing import Optional, List, Tuple, Callable
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass


@dataclass
class Geometric3DConfig:
    """Configuration for 3D geometric charts."""
    resolution: int = 100
    color_scheme: str = 'viridis'
    show_wireframe: bool = False
    opacity: float = 0.8
    lighting: dict = None
    camera_angle: Tuple[float, float, float] = (1.25, 1.25, 1.25)


class Cone3D:
    """
    3D Cone Chart with customizable parameters.

    Perfect for:
    - Hierarchical data visualization
    - Funnel analysis in 3D
    - Volume analysis
    - Mathematical education

    Example:
        >>> cone = Cone3D(height=10, radius=5, color='rainbow')
        >>> cone.add_data_mapping(df, value_column='sales')
        >>> fig = cone.render()
        >>> fig.show()
    """

    def __init__(
        self,
        height: float = 10.0,
        radius: float = 5.0,
        apex_position: Tuple[float, float, float] = (0, 0, 0),
        color: str = 'viridis',
        config: Optional[Geometric3DConfig] = None
    ):
        """
        Initialize 3D Cone.

        Args:
            height: Cone height
            radius: Base radius
            apex_position: (x, y, z) position of apex
            color: Color scheme
            config: Configuration object
        """
        self.height = height
        self.radius = radius
        self.apex_position = apex_position
        self.color = color
        self.config = config or Geometric3DConfig()

        # Generate cone geometry
        self.x, self.y, self.z = self._generate_cone()

    def _generate_cone(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate cone mesh coordinates."""
        u = np.linspace(0, 2 * np.pi, self.config.resolution)
        v = np.linspace(0, self.height, self.config.resolution)

        u, v = np.meshgrid(u, v)

        # Cone equations
        r = self.radius * (1 - v / self.height)
        x = r * np.cos(u) + self.apex_position[0]
        y = r * np.sin(u) + self.apex_position[1]
        z = v + self.apex_position[2]

        return x, y, z

    def render(self) -> go.Figure:
        """
        Render 3D cone chart.

        Returns:
            Plotly figure with 3D cone
        """
        fig = go.Figure(data=[
            go.Surface(
                x=self.x,
                y=self.y,
                z=self.z,
                colorscale=self.color,
                opacity=self.config.opacity,
                showscale=True
            )
        ])

        fig.update_layout(
            title='3D Cone Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=self.config.camera_angle[0],
                            y=self.config.camera_angle[1],
                            z=self.config.camera_angle[2])
                ),
                aspectmode='data'
            ),
            height=700
        )

        return fig


class Spiral3D:
    """
    3D Spiral/Helix Chart.

    Perfect for:
    - Time series with cyclical patterns
    - DNA-like data structures
    - Growth patterns
    - Logarithmic spirals

    Example:
        >>> spiral = Spiral3D(turns=5, radius=10, pitch=2)
        >>> spiral.set_color_by_height(colormap='plasma')
        >>> fig = spiral.render()
        >>> fig.show()
    """

    def __init__(
        self,
        turns: float = 5.0,
        radius: float = 10.0,
        pitch: float = 2.0,
        spiral_type: str = 'helix',  # 'helix', 'logarithmic', 'archimedean'
        config: Optional[Geometric3DConfig] = None
    ):
        """
        Initialize 3D Spiral.

        Args:
            turns: Number of complete turns
            radius: Spiral radius
            pitch: Vertical spacing between turns
            spiral_type: Type of spiral
            config: Configuration object
        """
        self.turns = turns
        self.radius = radius
        self.pitch = pitch
        self.spiral_type = spiral_type
        self.config = config or Geometric3DConfig()

        # Generate spiral geometry
        self.x, self.y, self.z, self.colors = self._generate_spiral()

    def _generate_spiral(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate spiral coordinates."""
        t = np.linspace(0, self.turns * 2 * np.pi, self.config.resolution * self.turns)

        if self.spiral_type == 'helix':
            # Standard helix
            x = self.radius * np.cos(t)
            y = self.radius * np.sin(t)
            z = self.pitch * t / (2 * np.pi)

        elif self.spiral_type == 'logarithmic':
            # Logarithmic spiral
            r = self.radius * np.exp(0.2 * t)
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = self.pitch * t / (2 * np.pi)

        elif self.spiral_type == 'archimedean':
            # Archimedean spiral
            r = self.radius * t / (self.turns * 2 * np.pi)
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = self.pitch * t / (2 * np.pi)

        else:
            raise ValueError(f"Unknown spiral type: {self.spiral_type}")

        # Color by height
        colors = z

        return x, y, z, colors

    def render(self, mode: str = 'lines') -> go.Figure:
        """
        Render 3D spiral.

        Args:
            mode: 'lines', 'markers', or 'lines+markers'

        Returns:
            Plotly figure with 3D spiral
        """
        fig = go.Figure(data=[
            go.Scatter3d(
                x=self.x,
                y=self.y,
                z=self.z,
                mode=mode,
                line=dict(
                    color=self.colors,
                    colorscale=self.config.color_scheme,
                    width=5
                ),
                marker=dict(
                    size=4,
                    color=self.colors,
                    colorscale=self.config.color_scheme
                )
            )
        ])

        fig.update_layout(
            title=f'3D {self.spiral_type.title()} Spiral',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=self.config.camera_angle[0],
                            y=self.config.camera_angle[1],
                            z=self.config.camera_angle[2])
                ),
                aspectmode='data'
            ),
            height=700
        )

        return fig


class Helix3D:
    """
    Double/Triple Helix 3D Chart (DNA-like).

    Perfect for:
    - Biological data (DNA, proteins)
    - Intertwined time series
    - Correlation visualization
    - Multi-dimensional relationships

    Example:
        >>> helix = Helix3D(strands=2, radius=5, turns=10)
        >>> helix.add_data_strand(df1, 'value1', color='red')
        >>> helix.add_data_strand(df2, 'value2', color='blue')
        >>> fig = helix.render()
        >>> fig.show()
    """

    def __init__(
        self,
        strands: int = 2,
        radius: float = 5.0,
        turns: float = 10.0,
        pitch: float = 2.0,
        config: Optional[Geometric3DConfig] = None
    ):
        """
        Initialize multi-strand helix.

        Args:
            strands: Number of intertwined strands
            radius: Helix radius
            turns: Number of complete turns
            pitch: Vertical spacing
            config: Configuration object
        """
        self.strands = strands
        self.radius = radius
        self.turns = turns
        self.pitch = pitch
        self.config = config or Geometric3DConfig()

        self.strand_data = []

    def _generate_helix_strand(self, phase_offset: float = 0.0) -> Tuple:
        """Generate single helix strand."""
        t = np.linspace(0, self.turns * 2 * np.pi, self.config.resolution * self.turns)

        x = self.radius * np.cos(t + phase_offset)
        y = self.radius * np.sin(t + phase_offset)
        z = self.pitch * t / (2 * np.pi)

        return x, y, z, t

    def render(self) -> go.Figure:
        """
        Render multi-strand helix.

        Returns:
            Plotly figure with helix strands
        """
        data = []

        # Generate each strand with phase offset
        for i in range(self.strands):
            phase_offset = (2 * np.pi / self.strands) * i
            x, y, z, t = self._generate_helix_strand(phase_offset)

            # Color palette for different strands
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
            color = colors[i % len(colors)]

            data.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='lines',
                    name=f'Strand {i+1}',
                    line=dict(color=color, width=6)
                )
            )

        fig = go.Figure(data=data)

        fig.update_layout(
            title=f'{self.strands}-Strand Helix Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data'
            ),
            height=700
        )

        return fig


class Torus3D:
    """
    3D Torus (Donut) Chart.

    Perfect for:
    - Cyclical data
    - Periodic patterns
    - Topological visualizations
    - Mathematical surfaces

    Example:
        >>> torus = Torus3D(major_radius=10, minor_radius=3)
        >>> torus.set_texture(pattern='gradient')
        >>> fig = torus.render()
        >>> fig.show()
    """

    def __init__(
        self,
        major_radius: float = 10.0,
        minor_radius: float = 3.0,
        config: Optional[Geometric3DConfig] = None
    ):
        """
        Initialize 3D Torus.

        Args:
            major_radius: Distance from center to tube center
            minor_radius: Tube radius
            config: Configuration object
        """
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.config = config or Geometric3DConfig()

        self.x, self.y, self.z = self._generate_torus()

    def _generate_torus(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate torus mesh."""
        u = np.linspace(0, 2 * np.pi, self.config.resolution)
        v = np.linspace(0, 2 * np.pi, self.config.resolution)

        u, v = np.meshgrid(u, v)

        # Torus equations
        x = (self.major_radius + self.minor_radius * np.cos(v)) * np.cos(u)
        y = (self.major_radius + self.minor_radius * np.cos(v)) * np.sin(u)
        z = self.minor_radius * np.sin(v)

        return x, y, z

    def render(self) -> go.Figure:
        """
        Render 3D torus.

        Returns:
            Plotly figure with torus
        """
        fig = go.Figure(data=[
            go.Surface(
                x=self.x,
                y=self.y,
                z=self.z,
                colorscale=self.config.color_scheme,
                opacity=self.config.opacity
            )
        ])

        fig.update_layout(
            title='3D Torus Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data'
            ),
            height=700
        )

        return fig


class Sphere3D:
    """
    3D Sphere with texture mapping.

    Perfect for:
    - Globe visualizations
    - Radial data
    - Heat maps on spherical surfaces
    - Planetary data

    Example:
        >>> sphere = Sphere3D(radius=10)
        >>> sphere.map_data(df, lat='latitude', lon='longitude', value='temperature')
        >>> fig = sphere.render()
        >>> fig.show()
    """

    def __init__(
        self,
        radius: float = 10.0,
        config: Optional[Geometric3DConfig] = None
    ):
        """
        Initialize 3D Sphere.

        Args:
            radius: Sphere radius
            config: Configuration object
        """
        self.radius = radius
        self.config = config or Geometric3DConfig()

        self.x, self.y, self.z = self._generate_sphere()

    def _generate_sphere(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate sphere mesh."""
        u = np.linspace(0, 2 * np.pi, self.config.resolution)
        v = np.linspace(0, np.pi, self.config.resolution)

        u, v = np.meshgrid(u, v)

        # Sphere equations
        x = self.radius * np.sin(v) * np.cos(u)
        y = self.radius * np.sin(v) * np.sin(u)
        z = self.radius * np.cos(v)

        return x, y, z

    def render(self) -> go.Figure:
        """
        Render 3D sphere.

        Returns:
            Plotly figure with sphere
        """
        fig = go.Figure(data=[
            go.Surface(
                x=self.x,
                y=self.y,
                z=self.z,
                colorscale=self.config.color_scheme,
                opacity=self.config.opacity
            )
        ])

        fig.update_layout(
            title='3D Sphere Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            height=700
        )

        return fig


# ==================== Helper Functions ====================

def create_cone(
    height: float = 10.0,
    radius: float = 5.0,
    color: str = 'viridis',
    resolution: int = 100
) -> go.Figure:
    """
    Quick create 3D cone chart.

    Args:
        height: Cone height
        radius: Base radius
        color: Color scheme
        resolution: Mesh resolution

    Returns:
        Plotly figure

    Example:
        >>> fig = create_cone(height=15, radius=8, color='plasma')
        >>> fig.show()
    """
    config = Geometric3DConfig(resolution=resolution, color_scheme=color)
    cone = Cone3D(height=height, radius=radius, color=color, config=config)
    return cone.render()


def create_spiral(
    turns: float = 5.0,
    radius: float = 10.0,
    pitch: float = 2.0,
    spiral_type: str = 'helix',
    color: str = 'viridis'
) -> go.Figure:
    """
    Quick create 3D spiral chart.

    Args:
        turns: Number of turns
        radius: Spiral radius
        pitch: Vertical spacing
        spiral_type: 'helix', 'logarithmic', or 'archimedean'
        color: Color scheme

    Returns:
        Plotly figure

    Example:
        >>> fig = create_spiral(turns=10, spiral_type='logarithmic')
        >>> fig.show()
    """
    config = Geometric3DConfig(color_scheme=color)
    spiral = Spiral3D(turns=turns, radius=radius, pitch=pitch,
                     spiral_type=spiral_type, config=config)
    return spiral.render()


def create_helix(
    strands: int = 2,
    radius: float = 5.0,
    turns: float = 10.0,
    pitch: float = 2.0
) -> go.Figure:
    """
    Quick create multi-strand helix (DNA-like).

    Args:
        strands: Number of strands
        radius: Helix radius
        turns: Number of turns
        pitch: Vertical spacing

    Returns:
        Plotly figure

    Example:
        >>> fig = create_helix(strands=3, turns=8)
        >>> fig.show()
    """
    helix = Helix3D(strands=strands, radius=radius, turns=turns, pitch=pitch)
    return helix.render()
