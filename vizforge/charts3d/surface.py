"""
VizForge 3D Surface Charts

Advanced 3D surface visualizations: parametric, implicit, mesh surfaces.
Part of VizForge v1.1.0 - Super AGI 3D Features.
"""

from typing import Optional, List, Tuple, Callable, Union
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
import pandas as pd


@dataclass
class Surface3DConfig:
    """Configuration for 3D surface charts."""
    resolution: int = 100
    color_scheme: str = 'viridis'
    show_contours: bool = True
    contour_color: str = 'white'
    opacity: float = 0.9
    lighting: dict = None
    camera_angle: Tuple[float, float, float] = (1.25, 1.25, 1.25)
    wireframe: bool = False


class Surface3D:
    """
    Basic 3D Surface Chart.

    Perfect for:
    - Mathematical function visualization
    - Elevation maps
    - Heatmap surfaces
    - Scientific data

    Example:
        >>> # Create surface from function
        >>> def f(x, y):
        ...     return np.sin(np.sqrt(x**2 + y**2))
        >>> surf = Surface3D(f, x_range=(-5, 5), y_range=(-5, 5))
        >>> fig = surf.render()
        >>> fig.show()
    """

    def __init__(
        self,
        function: Optional[Callable] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        x_range: Tuple[float, float] = (-10, 10),
        y_range: Tuple[float, float] = (-10, 10),
        config: Optional[Surface3DConfig] = None
    ):
        """
        Initialize 3D Surface.

        Args:
            function: Z = f(x, y) callable
            data: Pre-computed Z values (2D array or DataFrame)
            x_range: (min, max) for x-axis
            y_range: (min, max) for y-axis
            config: Configuration object
        """
        self.function = function
        self.data = data
        self.x_range = x_range
        self.y_range = y_range
        self.config = config or Surface3DConfig()

        # Generate surface
        self.x, self.y, self.z = self._generate_surface()

    def _generate_surface(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate surface mesh."""
        x = np.linspace(self.x_range[0], self.x_range[1], self.config.resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], self.config.resolution)

        x, y = np.meshgrid(x, y)

        if self.function is not None:
            # Compute from function
            z = self.function(x, y)
        elif self.data is not None:
            # Use provided data
            if isinstance(self.data, pd.DataFrame):
                z = self.data.values
            else:
                z = self.data
        else:
            raise ValueError("Either function or data must be provided")

        return x, y, z

    def render(self, show_wireframe: bool = None) -> go.Figure:
        """
        Render 3D surface.

        Args:
            show_wireframe: Override config wireframe setting

        Returns:
            Plotly figure with 3D surface
        """
        wireframe = show_wireframe if show_wireframe is not None else self.config.wireframe

        # Create surface trace
        surface = go.Surface(
            x=self.x,
            y=self.y,
            z=self.z,
            colorscale=self.config.color_scheme,
            opacity=self.config.opacity,
            showscale=True,
            contours={
                'z': {
                    'show': self.config.show_contours,
                    'usecolormap': True,
                    'highlightcolor': self.config.contour_color,
                    'project': {'z': True}
                }
            } if self.config.show_contours else {}
        )

        data = [surface]

        # Add wireframe if requested
        if wireframe:
            wireframe_trace = go.Scatter3d(
                x=self.x.flatten(),
                y=self.y.flatten(),
                z=self.z.flatten(),
                mode='markers',
                marker=dict(size=1, color='black'),
                showlegend=False
            )
            data.append(wireframe_trace)

        fig = go.Figure(data=data)

        fig.update_layout(
            title='3D Surface Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(
                        x=self.config.camera_angle[0],
                        y=self.config.camera_angle[1],
                        z=self.config.camera_angle[2]
                    )
                ),
                aspectmode='auto'
            ),
            height=700
        )

        return fig


class ParametricSurface3D:
    """
    3D Parametric Surface Chart.

    Perfect for:
    - Complex mathematical surfaces
    - Klein bottles, Mobius strips
    - Custom 3D shapes
    - Physics simulations

    Example:
        >>> # Mobius strip
        >>> def mobius(u, v):
        ...     x = (1 + v/2 * np.cos(u/2)) * np.cos(u)
        ...     y = (1 + v/2 * np.cos(u/2)) * np.sin(u)
        ...     z = v/2 * np.sin(u/2)
        ...     return x, y, z
        >>> surf = ParametricSurface3D(mobius, u_range=(0, 2*np.pi), v_range=(-1, 1))
        >>> fig = surf.render()
        >>> fig.show()
    """

    def __init__(
        self,
        parametric_func: Callable,
        u_range: Tuple[float, float] = (0, 2 * np.pi),
        v_range: Tuple[float, float] = (0, 2 * np.pi),
        config: Optional[Surface3DConfig] = None
    ):
        """
        Initialize parametric surface.

        Args:
            parametric_func: Function (u, v) -> (x, y, z)
            u_range: (min, max) for u parameter
            v_range: (min, max) for v parameter
            config: Configuration object
        """
        self.parametric_func = parametric_func
        self.u_range = u_range
        self.v_range = v_range
        self.config = config or Surface3DConfig()

        # Generate surface
        self.x, self.y, self.z = self._generate_parametric_surface()

    def _generate_parametric_surface(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate parametric surface mesh."""
        u = np.linspace(self.u_range[0], self.u_range[1], self.config.resolution)
        v = np.linspace(self.v_range[0], self.v_range[1], self.config.resolution)

        u, v = np.meshgrid(u, v)

        # Evaluate parametric function
        x, y, z = self.parametric_func(u, v)

        return x, y, z

    def render(self) -> go.Figure:
        """
        Render parametric surface.

        Returns:
            Plotly figure with parametric surface
        """
        fig = go.Figure(data=[
            go.Surface(
                x=self.x,
                y=self.y,
                z=self.z,
                colorscale=self.config.color_scheme,
                opacity=self.config.opacity,
                showscale=True
            )
        ])

        fig.update_layout(
            title='3D Parametric Surface',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(
                        x=self.config.camera_angle[0],
                        y=self.config.camera_angle[1],
                        z=self.config.camera_angle[2]
                    )
                ),
                aspectmode='data'
            ),
            height=700
        )

        return fig


class ImplicitSurface3D:
    """
    3D Implicit Surface Chart (Isosurface).

    Perfect for:
    - Equation surfaces: f(x,y,z) = c
    - 3D contours
    - Level sets
    - Medical imaging (MRI/CT)

    Example:
        >>> # Sphere: x^2 + y^2 + z^2 = 1
        >>> def sphere_func(x, y, z):
        ...     return x**2 + y**2 + z**2
        >>> surf = ImplicitSurface3D(sphere_func, level=1.0)
        >>> fig = surf.render()
        >>> fig.show()
    """

    def __init__(
        self,
        implicit_func: Callable,
        level: float = 0.0,
        bounds: Tuple[float, float] = (-5, 5),
        config: Optional[Surface3DConfig] = None
    ):
        """
        Initialize implicit surface.

        Args:
            implicit_func: Function f(x, y, z)
            level: Isosurface level (f = level)
            bounds: (min, max) for all axes
            config: Configuration object
        """
        self.implicit_func = implicit_func
        self.level = level
        self.bounds = bounds
        self.config = config or Surface3DConfig()

        # Generate volume and extract isosurface
        self.vertices, self.faces = self._extract_isosurface()

    def _extract_isosurface(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract isosurface using marching cubes algorithm.

        Note: Simplified implementation. For production, use scikit-image.
        """
        # Create 3D grid
        n = self.config.resolution
        x = np.linspace(self.bounds[0], self.bounds[1], n)
        y = np.linspace(self.bounds[0], self.bounds[1], n)
        z = np.linspace(self.bounds[0], self.bounds[1], n)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Evaluate function on grid
        volume = self.implicit_func(X, Y, Z)

        # Simplified isosurface extraction (use marching cubes for production)
        # For now, create a simple mesh representation
        vertices = np.array([[0, 0, 0]])  # Placeholder
        faces = np.array([[0, 0, 0]])     # Placeholder

        return vertices, faces

    def render(self) -> go.Figure:
        """
        Render implicit surface.

        Returns:
            Plotly figure with isosurface
        """
        # Create 3D grid for visualization
        n = self.config.resolution
        x = np.linspace(self.bounds[0], self.bounds[1], n)
        y = np.linspace(self.bounds[0], self.bounds[1], n)
        z = np.linspace(self.bounds[0], self.bounds[1], n)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        values = self.implicit_func(X, Y, Z)

        # Use go.Isosurface
        fig = go.Figure(data=[
            go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=values.flatten(),
                isomin=self.level - 0.1,
                isomax=self.level + 0.1,
                surface_count=1,
                colorscale=self.config.color_scheme,
                opacity=self.config.opacity,
                caps=dict(x_show=False, y_show=False, z_show=False)
            )
        ])

        fig.update_layout(
            title=f'3D Implicit Surface (f = {self.level})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(
                        x=self.config.camera_angle[0],
                        y=self.config.camera_angle[1],
                        z=self.config.camera_angle[2]
                    )
                ),
                aspectmode='cube'
            ),
            height=700
        )

        return fig


class MeshSurface3D:
    """
    3D Mesh Surface from vertex and face data.

    Perfect for:
    - 3D model visualization
    - CAD/CAM data
    - Topology visualization
    - Point cloud surfaces

    Example:
        >>> # Load 3D mesh from file
        >>> vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
        >>> faces = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
        >>> mesh = MeshSurface3D(vertices, faces)
        >>> fig = mesh.render()
        >>> fig.show()
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_colors: Optional[np.ndarray] = None,
        config: Optional[Surface3DConfig] = None
    ):
        """
        Initialize mesh surface.

        Args:
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face indices
            vertex_colors: Optional Nx1 array of vertex colors
            config: Configuration object
        """
        self.vertices = vertices
        self.faces = faces
        self.vertex_colors = vertex_colors
        self.config = config or Surface3DConfig()

    def render(self) -> go.Figure:
        """
        Render 3D mesh.

        Returns:
            Plotly figure with 3D mesh
        """
        # Extract coordinates
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        z = self.vertices[:, 2]

        # Extract face indices
        i = self.faces[:, 0]
        j = self.faces[:, 1]
        k = self.faces[:, 2]

        mesh_trace = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            colorscale=self.config.color_scheme,
            intensity=self.vertex_colors if self.vertex_colors is not None else z,
            opacity=self.config.opacity,
            showscale=True
        )

        fig = go.Figure(data=[mesh_trace])

        fig.update_layout(
            title='3D Mesh Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(
                        x=self.config.camera_angle[0],
                        y=self.config.camera_angle[1],
                        z=self.config.camera_angle[2]
                    )
                ),
                aspectmode='data'
            ),
            height=700
        )

        return fig

    @classmethod
    def from_point_cloud(
        cls,
        points: np.ndarray,
        method: str = 'delaunay'
    ) -> 'MeshSurface3D':
        """
        Create mesh from point cloud.

        Args:
            points: Nx3 array of points
            method: 'delaunay' or 'alpha_shape'

        Returns:
            MeshSurface3D instance
        """
        from scipy.spatial import Delaunay

        if method == 'delaunay':
            # 3D Delaunay triangulation
            tri = Delaunay(points)
            vertices = points
            faces = tri.simplices
        else:
            raise ValueError(f"Unknown method: {method}")

        return cls(vertices, faces)


# ==================== Helper Functions ====================

def create_surface(
    function: Callable,
    x_range: Tuple[float, float] = (-10, 10),
    y_range: Tuple[float, float] = (-10, 10),
    color: str = 'viridis',
    resolution: int = 100
) -> go.Figure:
    """
    Quick create 3D surface chart.

    Args:
        function: Z = f(x, y) callable
        x_range: (min, max) for x-axis
        y_range: (min, max) for y-axis
        color: Color scheme
        resolution: Mesh resolution

    Returns:
        Plotly figure

    Example:
        >>> fig = create_surface(lambda x, y: np.sin(x) * np.cos(y))
        >>> fig.show()
    """
    config = Surface3DConfig(resolution=resolution, color_scheme=color)
    surf = Surface3D(function=function, x_range=x_range, y_range=y_range, config=config)
    return surf.render()


def create_parametric_surface(
    parametric_func: Callable,
    u_range: Tuple[float, float] = (0, 2 * np.pi),
    v_range: Tuple[float, float] = (0, 2 * np.pi),
    color: str = 'viridis'
) -> go.Figure:
    """
    Quick create parametric surface.

    Args:
        parametric_func: Function (u, v) -> (x, y, z)
        u_range: (min, max) for u parameter
        v_range: (min, max) for v parameter
        color: Color scheme

    Returns:
        Plotly figure

    Example:
        >>> # Torus
        >>> def torus(u, v):
        ...     R, r = 2, 1
        ...     x = (R + r*np.cos(v)) * np.cos(u)
        ...     y = (R + r*np.cos(v)) * np.sin(u)
        ...     z = r * np.sin(v)
        ...     return x, y, z
        >>> fig = create_parametric_surface(torus)
        >>> fig.show()
    """
    config = Surface3DConfig(color_scheme=color)
    surf = ParametricSurface3D(parametric_func, u_range, v_range, config)
    return surf.render()


# ==================== Pre-built Parametric Surfaces ====================

def mobius_strip() -> go.Figure:
    """Create Mobius strip surface."""
    def mobius(u, v):
        u = u * 2 * np.pi
        x = (1 + v/2 * np.cos(u/2)) * np.cos(u)
        y = (1 + v/2 * np.cos(u/2)) * np.sin(u)
        z = v/2 * np.sin(u/2)
        return x, y, z

    return create_parametric_surface(mobius, u_range=(0, 1), v_range=(-1, 1))


def klein_bottle() -> go.Figure:
    """Create Klein bottle surface."""
    def klein(u, v):
        u = u * 2 * np.pi
        v = v * 2 * np.pi
        r = 4 * (1 - np.cos(u) / 2)

        x = 6 * np.cos(u) * (1 + np.sin(u)) + r * np.cos(v + np.pi)
        y = 16 * np.sin(u)
        z = r * np.sin(v)

        return x, y, z

    return create_parametric_surface(klein, u_range=(0, 1), v_range=(0, 1))


def seashell() -> go.Figure:
    """Create seashell (conch) surface."""
    def shell(u, v):
        u = u * 2 * np.pi
        v = v * 2 * np.pi

        x = 2 * (1 - v / (2 * np.pi)) * np.cos(2 * v) * (1 + np.cos(u))
        y = 2 * (1 - v / (2 * np.pi)) * np.sin(2 * v) * (1 + np.cos(u))
        z = v / (2 * np.pi) + 2 * (1 - v / (2 * np.pi)) * np.sin(u)

        return x, y, z

    return create_parametric_surface(shell, u_range=(0, 1), v_range=(0, 1))
