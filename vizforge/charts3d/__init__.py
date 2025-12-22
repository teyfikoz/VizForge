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

from .geometric import (
    Cone3D,
    Spiral3D,
    Helix3D,
    Torus3D,
    Sphere3D,
    create_cone,
    create_spiral,
    create_helix
)

from .surface import (
    Surface3D,
    ParametricSurface3D,
    ImplicitSurface3D,
    MeshSurface3D,
    create_surface,
    create_parametric_surface,
    mobius_strip,
    klein_bottle,
    seashell
)

from .scientific import (
    VectorField3D,
    Isosurface3D,
    VolumeRender3D,
    MolecularStructure3D,
    create_vector_field,
    create_isosurface,
    magnetic_dipole_field,
    hydrogen_orbital
)

from .controls import (
    RotationControl,
    CameraControl,
    Animation3D,
    VRController,
    enable_360_rotation,
    enable_vr_mode,
    add_viewpoint_buttons
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
