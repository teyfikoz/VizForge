"""
VizForge Animations Module

Modern animations, transitions, and gestures for stunning visualizations.
Part of VizForge v1.0.0 - Super AGI features.
"""

from .easing import (
    EasingFunction,
    bezier,
    ease_in,
    ease_in_bounce,
    ease_in_cubic,
    ease_in_elastic,
    ease_in_out,
    ease_in_out_cubic,
    ease_out,
    ease_out_bounce,
    ease_out_cubic,
    ease_out_elastic,
    linear,
)
from .gestures import (
    GestureConfig,
    GestureHandler,
    GestureType,
    TouchEvent,
    enable_mobile_gestures,
    setup_gestures,
)
from .transitions import (
    AnimationEngine,
    TransitionConfig,
    TransitionType,
    apply_transition,
    create_transition,
)

__all__ = [
    # Transitions
    'TransitionConfig',
    'TransitionType',
    'AnimationEngine',
    'apply_transition',
    'create_transition',

    # Easing Functions
    'EasingFunction',
    'linear',
    'ease_in',
    'ease_out',
    'ease_in_out',
    'ease_in_cubic',
    'ease_out_cubic',
    'ease_in_out_cubic',
    'ease_in_elastic',
    'ease_out_elastic',
    'ease_in_bounce',
    'ease_out_bounce',
    'bezier',

    # Gestures
    'GestureType',
    'GestureConfig',
    'GestureHandler',
    'TouchEvent',
    'setup_gestures',
    'enable_mobile_gestures',
]
