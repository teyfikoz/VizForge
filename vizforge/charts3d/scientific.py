"""
VizForge 3D Scientific Visualizations

Advanced scientific 3D charts: vector fields, isosurfaces, volume rendering, molecular structures.
Part of VizForge v1.1.0 - Super AGI 3D Features.
"""

from typing import Optional, List, Tuple, Callable, Union
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
import pandas as pd


@dataclass
class Scientific3DConfig:
    """Configuration for 3D scientific charts."""
    resolution: int = 20
    color_scheme: str = 'viridis'
    opacity: float = 0.8
    arrow_scale: float = 1.0
    camera_angle: Tuple[float, float, float] = (1.5, 1.5, 1.5)


class VectorField3D:
    """
    3D Vector Field Visualization.

    Perfect for:
    - Electromagnetic fields
    - Fluid dynamics
    - Gradient fields
    - Force fields

    Example:
        >>> # Magnetic field
        >>> def field(x, y, z):
        ...     # Returns (vx, vy, vz) at each point
        ...     return -y, x, 0
        >>> vf = VectorField3D(field, bounds=(-5, 5))
        >>> fig = vf.render()
        >>> fig.show()
    """

    def __init__(
        self,
        vector_func: Callable,
        bounds: Tuple[float, float] = (-5, 5),
        config: Optional[Scientific3DConfig] = None
    ):
        """
        Initialize vector field.

        Args:
            vector_func: Function (x, y, z) -> (vx, vy, vz)
            bounds: (min, max) for all axes
            config: Configuration object
        """
        self.vector_func = vector_func
        self.bounds = bounds
        self.config = config or Scientific3DConfig()

        # Generate vector field
        self.points, self.vectors = self._generate_vector_field()

    def _generate_vector_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate vector field data."""
        n = self.config.resolution
        x = np.linspace(self.bounds[0], self.bounds[1], n)
        y = np.linspace(self.bounds[0], self.bounds[1], n)
        z = np.linspace(self.bounds[0], self.bounds[1], n)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Evaluate vector function
        vx, vy, vz = self.vector_func(X, Y, Z)

        # Flatten for plotting
        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        vectors = np.column_stack([vx.flatten(), vy.flatten(), vz.flatten()])

        return points, vectors

    def render(self, mode: str = 'cone') -> go.Figure:
        """
        Render vector field.

        Args:
            mode: 'cone', 'arrow', or 'streamline'

        Returns:
            Plotly figure with vector field
        """
        if mode == 'cone':
            # Use cone plot
            fig = go.Figure(data=[
                go.Cone(
                    x=self.points[:, 0],
                    y=self.points[:, 1],
                    z=self.points[:, 2],
                    u=self.vectors[:, 0],
                    v=self.vectors[:, 1],
                    w=self.vectors[:, 2],
                    colorscale=self.config.color_scheme,
                    sizemode='absolute',
                    sizeref=self.config.arrow_scale,
                    showscale=True
                )
            ])

        elif mode == 'arrow':
            # Use arrows (quiver plot)
            # Create arrow traces
            data = []
            for i in range(0, len(self.points), 10):  # Subsample for clarity
                p = self.points[i]
                v = self.vectors[i] * self.config.arrow_scale

                # Arrow line
                data.append(
                    go.Scatter3d(
                        x=[p[0], p[0] + v[0]],
                        y=[p[1], p[1] + v[1]],
                        z=[p[2], p[2] + v[2]],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        showlegend=False
                    )
                )

            fig = go.Figure(data=data)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        fig.update_layout(
            title='3D Vector Field Visualization',
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


class Isosurface3D:
    """
    3D Isosurface (Level Set) Visualization.

    Perfect for:
    - Medical imaging (MRI, CT)
    - Molecular orbitals
    - Density visualization
    - Scalar field analysis

    Example:
        >>> # Electron density
        >>> def density(x, y, z):
        ...     r = np.sqrt(x**2 + y**2 + z**2)
        ...     return np.exp(-r)
        >>> iso = Isosurface3D(density, levels=[0.1, 0.3, 0.5])
        >>> fig = iso.render()
        >>> fig.show()
    """

    def __init__(
        self,
        scalar_func: Callable,
        levels: List[float],
        bounds: Tuple[float, float] = (-5, 5),
        config: Optional[Scientific3DConfig] = None
    ):
        """
        Initialize isosurface.

        Args:
            scalar_func: Scalar field function f(x, y, z)
            levels: List of isosurface levels
            bounds: (min, max) for all axes
            config: Configuration object
        """
        self.scalar_func = scalar_func
        self.levels = levels
        self.bounds = bounds
        self.config = config or Scientific3DConfig()

        # Generate volume data
        self.X, self.Y, self.Z, self.values = self._generate_volume()

    def _generate_volume(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate 3D volume data."""
        n = self.config.resolution * 2  # Higher resolution for smooth isosurfaces
        x = np.linspace(self.bounds[0], self.bounds[1], n)
        y = np.linspace(self.bounds[0], self.bounds[1], n)
        z = np.linspace(self.bounds[0], self.bounds[1], n)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Evaluate scalar function
        values = self.scalar_func(X, Y, Z)

        return X, Y, Z, values

    def render(self) -> go.Figure:
        """
        Render isosurfaces.

        Returns:
            Plotly figure with isosurfaces
        """
        data = []

        # Create isosurface for each level
        for i, level in enumerate(self.levels):
            iso = go.Isosurface(
                x=self.X.flatten(),
                y=self.Y.flatten(),
                z=self.Z.flatten(),
                value=self.values.flatten(),
                isomin=level - 0.01,
                isomax=level + 0.01,
                surface_count=1,
                colorscale=self.config.color_scheme,
                opacity=self.config.opacity,
                name=f'Level {level}',
                caps=dict(x_show=False, y_show=False, z_show=False)
            )
            data.append(iso)

        fig = go.Figure(data=data)

        fig.update_layout(
            title='3D Isosurface Visualization',
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


class VolumeRender3D:
    """
    3D Volume Rendering.

    Perfect for:
    - Medical imaging (CT, MRI)
    - Scientific data volumes
    - Density clouds
    - 3D scalar fields

    Example:
        >>> # Load medical scan
        >>> volume_data = np.random.rand(50, 50, 50)
        >>> vol = VolumeRender3D(volume_data)
        >>> fig = vol.render()
        >>> fig.show()
    """

    def __init__(
        self,
        volume: np.ndarray,
        config: Optional[Scientific3DConfig] = None
    ):
        """
        Initialize volume rendering.

        Args:
            volume: 3D numpy array of scalar values
            config: Configuration object
        """
        self.volume = volume
        self.config = config or Scientific3DConfig()

    def render(self, opacity_scale: float = 0.1) -> go.Figure:
        """
        Render volume.

        Args:
            opacity_scale: Opacity scaling factor

        Returns:
            Plotly figure with volume rendering
        """
        # Get volume dimensions
        nz, ny, nx = self.volume.shape

        # Create coordinate grids
        X, Y, Z = np.mgrid[0:nx, 0:ny, 0:nz]

        fig = go.Figure(data=[
            go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=self.volume.flatten(),
                isomin=self.volume.min(),
                isomax=self.volume.max(),
                opacity=opacity_scale,
                surface_count=17,  # More surfaces = smoother rendering
                colorscale=self.config.color_scheme,
                caps=dict(x_show=False, y_show=False, z_show=False)
            )
        ])

        fig.update_layout(
            title='3D Volume Rendering',
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


class MolecularStructure3D:
    """
    3D Molecular Structure Visualization.

    Perfect for:
    - Chemistry (molecules)
    - Crystallography
    - Protein structures
    - Atomic simulations

    Example:
        >>> # Water molecule (H2O)
        >>> atoms = [
        ...     ('O', [0, 0, 0]),
        ...     ('H', [0.96, 0, 0]),
        ...     ('H', [-0.24, 0.93, 0])
        ... ]
        >>> bonds = [(0, 1), (0, 2)]
        >>> mol = MolecularStructure3D(atoms, bonds)
        >>> fig = mol.render()
        >>> fig.show()
    """

    # Atomic radii (Angstroms)
    ATOMIC_RADII = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66,
        'F': 0.57, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
        'default': 0.5
    }

    # Atomic colors (CPK coloring)
    ATOMIC_COLORS = {
        'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red',
        'F': 'green', 'P': 'orange', 'S': 'yellow', 'Cl': 'green',
        'default': 'pink'
    }

    def __init__(
        self,
        atoms: List[Tuple[str, List[float]]],
        bonds: Optional[List[Tuple[int, int]]] = None,
        config: Optional[Scientific3DConfig] = None
    ):
        """
        Initialize molecular structure.

        Args:
            atoms: List of (element, [x, y, z]) tuples
            bonds: List of (atom_i, atom_j) bond pairs
            config: Configuration object
        """
        self.atoms = atoms
        self.bonds = bonds or []
        self.config = config or Scientific3DConfig()

    def render(self, show_bonds: bool = True) -> go.Figure:
        """
        Render molecular structure.

        Args:
            show_bonds: Whether to show bonds

        Returns:
            Plotly figure with molecular structure
        """
        data = []

        # Draw atoms
        for element, pos in self.atoms:
            radius = self.ATOMIC_RADII.get(element, self.ATOMIC_RADII['default'])
            color = self.ATOMIC_COLORS.get(element, self.ATOMIC_COLORS['default'])

            # Create sphere for atom
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]

            atom_trace = go.Surface(
                x=x, y=y, z=z,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=0.9,
                name=element
            )
            data.append(atom_trace)

        # Draw bonds
        if show_bonds:
            for i, j in self.bonds:
                pos_i = self.atoms[i][1]
                pos_j = self.atoms[j][1]

                bond_trace = go.Scatter3d(
                    x=[pos_i[0], pos_j[0]],
                    y=[pos_i[1], pos_j[1]],
                    z=[pos_i[2], pos_j[2]],
                    mode='lines',
                    line=dict(color='gray', width=8),
                    showlegend=False
                )
                data.append(bond_trace)

        fig = go.Figure(data=data)

        fig.update_layout(
            title='Molecular Structure Visualization',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data'
            ),
            height=700,
            showlegend=False
        )

        return fig

    @classmethod
    def from_xyz(cls, xyz_string: str) -> 'MolecularStructure3D':
        """
        Create molecular structure from XYZ format string.

        Args:
            xyz_string: XYZ format molecular data

        Returns:
            MolecularStructure3D instance

        Example:
            >>> xyz = '''
            ... 3
            ... Water molecule
            ... O  0.000  0.000  0.000
            ... H  0.960  0.000  0.000
            ... H -0.240  0.930  0.000
            ... '''
            >>> mol = MolecularStructure3D.from_xyz(xyz)
        """
        lines = xyz_string.strip().split('\n')
        n_atoms = int(lines[0])

        atoms = []
        for line in lines[2:2+n_atoms]:
            parts = line.split()
            element = parts[0]
            coords = [float(x) for x in parts[1:4]]
            atoms.append((element, coords))

        # Auto-detect bonds (simple distance criterion)
        bonds = []
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                pos_i = np.array(atoms[i][1])
                pos_j = np.array(atoms[j][1])
                distance = np.linalg.norm(pos_i - pos_j)

                # Bond if distance < sum of radii * 1.2
                r_i = cls.ATOMIC_RADII.get(atoms[i][0], 0.5)
                r_j = cls.ATOMIC_RADII.get(atoms[j][0], 0.5)

                if distance < (r_i + r_j) * 1.5:
                    bonds.append((i, j))

        return cls(atoms, bonds)


# ==================== Helper Functions ====================

def create_vector_field(
    vector_func: Callable,
    bounds: Tuple[float, float] = (-5, 5),
    mode: str = 'cone',
    resolution: int = 10
) -> go.Figure:
    """
    Quick create vector field visualization.

    Args:
        vector_func: Function (x, y, z) -> (vx, vy, vz)
        bounds: (min, max) for all axes
        mode: 'cone' or 'arrow'
        resolution: Field resolution

    Returns:
        Plotly figure

    Example:
        >>> # Circular field
        >>> def field(x, y, z):
        ...     return -y, x, z*0
        >>> fig = create_vector_field(field)
        >>> fig.show()
    """
    config = Scientific3DConfig(resolution=resolution)
    vf = VectorField3D(vector_func, bounds, config)
    return vf.render(mode=mode)


def create_isosurface(
    scalar_func: Callable,
    levels: List[float],
    bounds: Tuple[float, float] = (-5, 5),
    color: str = 'viridis'
) -> go.Figure:
    """
    Quick create isosurface visualization.

    Args:
        scalar_func: Scalar field f(x, y, z)
        levels: Isosurface levels
        bounds: (min, max) for all axes
        color: Color scheme

    Returns:
        Plotly figure

    Example:
        >>> # Gaussian
        >>> def gaussian(x, y, z):
        ...     return np.exp(-(x**2 + y**2 + z**2))
        >>> fig = create_isosurface(gaussian, levels=[0.1, 0.5])
        >>> fig.show()
    """
    config = Scientific3DConfig(color_scheme=color)
    iso = Isosurface3D(scalar_func, levels, bounds, config)
    return iso.render()


# ==================== Pre-built Scientific Examples ====================

def magnetic_dipole_field() -> go.Figure:
    """Create magnetic dipole vector field."""
    def field(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.maximum(r, 0.1)  # Avoid singularity

        # Magnetic dipole oriented along z-axis
        m = 1.0  # Magnetic moment
        r3 = r**3
        r5 = r**5

        vx = 3 * m * x * z / r5
        vy = 3 * m * y * z / r5
        vz = m * (3 * z**2 - r**2) / r5

        return vx, vy, vz

    return create_vector_field(field, bounds=(-3, 3), mode='cone', resolution=8)


def hydrogen_orbital(n: int = 1, l: int = 0, m: int = 0) -> go.Figure:
    """
    Create hydrogen atomic orbital isosurface.

    Args:
        n: Principal quantum number
        l: Angular momentum quantum number
        m: Magnetic quantum number

    Returns:
        Plotly figure with orbital
    """
    def orbital_1s(x, y, z):
        """1s orbital."""
        r = np.sqrt(x**2 + y**2 + z**2)
        return np.exp(-r)

    def orbital_2s(x, y, z):
        """2s orbital."""
        r = np.sqrt(x**2 + y**2 + z**2)
        return (2 - r) * np.exp(-r/2)

    def orbital_2p(x, y, z):
        """2p orbital (m=0)."""
        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.maximum(r, 0.01)
        return z / r * np.exp(-r/2)

    # Select orbital
    if n == 1 and l == 0:
        func = orbital_1s
    elif n == 2 and l == 0:
        func = orbital_2s
    elif n == 2 and l == 1:
        func = orbital_2p
    else:
        func = orbital_1s

    return create_isosurface(func, levels=[0.1, 0.05], bounds=(-5, 5))
