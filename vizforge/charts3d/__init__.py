"""
VizForge 3D Charts Module

Advanced 3D visualizations with 360-degree rotation support.
Part of VizForge v1.1.0 - Super AGI 3D Features.

Features:
- 3D geometric charts (cone, spiral, helix, torus)
- 3D surface plots (parametric, implicit)
- 3D mesh visualizations
- 360-degree rotation controls
- VR/AR ready exports
- Scientific visualizations
"""

from .controls import (
    Animation3D,
    CameraControl,
    RotationControl,
    VRController,
    add_viewpoint_buttons,
    enable_360_rotation,
    enable_vr_mode,
)
from .geometric import (
    Cone3D,
    Helix3D,
    Sphere3D,
    Spiral3D,
    Torus3D,
    create_cone,
    create_helix,
    create_spiral,
)
from .scientific import (
    Isosurface3D,
    MolecularStructure3D,
    VectorField3D,
    VolumeRender3D,
    create_isosurface,
    create_vector_field,
    hydrogen_orbital,
    magnetic_dipole_field,
)
from .surface import (
    ImplicitSurface3D,
    MeshSurface3D,
    ParametricSurface3D,
    Surface3D,
    create_parametric_surface,
    create_surface,
    klein_bottle,
    mobius_strip,
    seashell,
)

__all__ = [
    # Geometric 3D
    'Cone3D',
    'Spiral3D',
    'Helix3D',
    'Torus3D',
    'Sphere3D',
    'create_cone',
    'create_spiral',
    'create_helix',

    # Surface 3D
    'Surface3D',
    'ParametricSurface3D',
    'ImplicitSurface3D',
    'MeshSurface3D',
    'create_surface',
    'create_parametric_surface',
    'mobius_strip',
    'klein_bottle',
    'seashell',

    # Scientific 3D
    'VectorField3D',
    'Isosurface3D',
    'VolumeRender3D',
    'MolecularStructure3D',
    'create_vector_field',
    'create_isosurface',
    'magnetic_dipole_field',
    'hydrogen_orbital',

    # Controls
    'RotationControl',
    'CameraControl',
    'Animation3D',
    'VRController',
    'enable_360_rotation',
    'enable_vr_mode',
    'add_viewpoint_buttons',
]
