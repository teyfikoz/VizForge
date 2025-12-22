"""
VizForge Easing Functions

Mathematical easing functions for smooth animations.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Callable
import math


# Type alias for easing functions
EasingFunction = Callable[[float], float]


# ==================== Linear ====================

def linear(t: float) -> float:
    """
    Linear easing (no acceleration).

    Args:
        t: Progress (0.0 to 1.0)

    Returns:
        Eased value (0.0 to 1.0)

    Example:
        >>> linear(0.5)
        0.5
    """
    return t


# ==================== Quadratic ====================

def ease_in(t: float) -> float:
    """
    Quadratic ease-in (accelerating from zero).

    Args:
        t: Progress (0.0 to 1.0)

    Returns:
        Eased value (0.0 to 1.0)

    Example:
        >>> ease_in(0.5)
        0.25
    """
    return t * t


def ease_out(t: float) -> float:
    """
    Quadratic ease-out (decelerating to zero).

    Args:
        t: Progress (0.0 to 1.0)

    Returns:
        Eased value (0.0 to 1.0)

    Example:
        >>> ease_out(0.5)
        0.75
    """
    return t * (2 - t)


def ease_in_out(t: float) -> float:
    """
    Quadratic ease-in-out (acceleration until halfway, then deceleration).

    Args:
        t: Progress (0.0 to 1.0)

    Returns:
        Eased value (0.0 to 1.0)

    Example:
        >>> ease_in_out(0.5)
        0.5
    """
    if t < 0.5:
        return 2 * t * t
    else:
        return -1 + (4 - 2 * t) * t


# ==================== Cubic ====================

def ease_in_cubic(t: float) -> float:
    """
    Cubic ease-in (strong acceleration from zero).

    Args:
        t: Progress (0.0 to 1.0)

    Returns:
        Eased value (0.0 to 1.0)
    """
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """
    Cubic ease-out (strong deceleration to zero).

    Args:
        t: Progress (0.0 to 1.0)

    Returns:
        Eased value (0.0 to 1.0)
    """
    t_shifted = t - 1
    return t_shifted * t_shifted * t_shifted + 1


def ease_in_out_cubic(t: float) -> float:
    """
    Cubic ease-in-out (strong acceleration/deceleration).

    Args:
        t: Progress (0.0 to 1.0)

    Returns:
        Eased value (0.0 to 1.0)
    """
    if t < 0.5:
        return 4 * t * t * t
    else:
        t_shifted = 2 * t - 2
        return (t_shifted * t_shifted * t_shifted + 2) / 2


# ==================== Elastic ====================

def ease_in_elastic(t: float, amplitude: float = 1.0, period: float = 0.3) -> float:
    """
    Elastic ease-in (elastic snap from zero).

    Creates a spring-like effect with overshoot.

    Args:
        t: Progress (0.0 to 1.0)
        amplitude: Amplitude of oscillation
        period: Period of oscillation

    Returns:
        Eased value (may overshoot 0.0-1.0 range)

    Example:
        >>> ease_in_elastic(0.5)
        -0.015625
    """
    if t == 0 or t == 1:
        return t

    s = period / 4
    t_shifted = t - 1
    return -(amplitude * math.pow(2, 10 * t_shifted) *
             math.sin((t_shifted - s) * (2 * math.pi) / period))


def ease_out_elastic(t: float, amplitude: float = 1.0, period: float = 0.3) -> float:
    """
    Elastic ease-out (elastic snap to target).

    Creates a spring-like effect with overshoot.

    Args:
        t: Progress (0.0 to 1.0)
        amplitude: Amplitude of oscillation
        period: Period of oscillation

    Returns:
        Eased value (may overshoot 0.0-1.0 range)

    Example:
        >>> ease_out_elastic(0.5)
        1.015625
    """
    if t == 0 or t == 1:
        return t

    s = period / 4
    return (amplitude * math.pow(2, -10 * t) *
            math.sin((t - s) * (2 * math.pi) / period) + 1)


def ease_in_out_elastic(t: float, amplitude: float = 1.0, period: float = 0.45) -> float:
    """
    Elastic ease-in-out (elastic snap both ways).

    Args:
        t: Progress (0.0 to 1.0)
        amplitude: Amplitude of oscillation
        period: Period of oscillation

    Returns:
        Eased value (may overshoot 0.0-1.0 range)
    """
    if t == 0 or t == 1:
        return t

    s = period / 4
    t_shifted = 2 * t - 1

    if t_shifted < 0:
        return -0.5 * (amplitude * math.pow(2, 10 * t_shifted) *
                      math.sin((t_shifted - s) * (2 * math.pi) / period))
    else:
        return 0.5 * (amplitude * math.pow(2, -10 * t_shifted) *
                     math.sin((t_shifted - s) * (2 * math.pi) / period)) + 1


# ==================== Bounce ====================

def ease_in_bounce(t: float) -> float:
    """
    Bounce ease-in (bouncing from zero).

    Args:
        t: Progress (0.0 to 1.0)

    Returns:
        Eased value (0.0 to 1.0)

    Example:
        >>> ease_in_bounce(0.5)
        0.234375
    """
    return 1 - ease_out_bounce(1 - t)


def ease_out_bounce(t: float) -> float:
    """
    Bounce ease-out (bouncing to target).

    Args:
        t: Progress (0.0 to 1.0)

    Returns:
        Eased value (0.0 to 1.0)

    Example:
        >>> ease_out_bounce(0.5)
        0.765625
    """
    if t < 1 / 2.75:
        return 7.5625 * t * t
    elif t < 2 / 2.75:
        t_shifted = t - 1.5 / 2.75
        return 7.5625 * t_shifted * t_shifted + 0.75
    elif t < 2.5 / 2.75:
        t_shifted = t - 2.25 / 2.75
        return 7.5625 * t_shifted * t_shifted + 0.9375
    else:
        t_shifted = t - 2.625 / 2.75
        return 7.5625 * t_shifted * t_shifted + 0.984375


def ease_in_out_bounce(t: float) -> float:
    """
    Bounce ease-in-out (bouncing both ways).

    Args:
        t: Progress (0.0 to 1.0)

    Returns:
        Eased value (0.0 to 1.0)
    """
    if t < 0.5:
        return ease_in_bounce(t * 2) * 0.5
    else:
        return ease_out_bounce(t * 2 - 1) * 0.5 + 0.5


# ==================== Bezier ====================

def bezier(p0: float, p1: float, p2: float, p3: float) -> EasingFunction:
    """
    Create cubic Bezier easing function.

    Args:
        p0: Control point 0 (0.0 to 1.0)
        p1: Control point 1 (0.0 to 1.0)
        p2: Control point 2 (0.0 to 1.0)
        p3: Control point 3 (0.0 to 1.0)

    Returns:
        Easing function

    Example:
        >>> ease_custom = bezier(0.25, 0.1, 0.25, 1.0)
        >>> ease_custom(0.5)
        0.5
    """
    def bezier_easing(t: float) -> float:
        """Cubic Bezier interpolation."""
        # Bezier formula: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        t_inv = 1 - t
        return (
            t_inv * t_inv * t_inv * p0 +
            3 * t_inv * t_inv * t * p1 +
            3 * t_inv * t * t * p2 +
            t * t * t * p3
        )

    return bezier_easing


# ==================== Common Bezier Presets ====================

def ease_in_sine(t: float) -> float:
    """Sine ease-in."""
    return 1 - math.cos(t * math.pi / 2)


def ease_out_sine(t: float) -> float:
    """Sine ease-out."""
    return math.sin(t * math.pi / 2)


def ease_in_out_sine(t: float) -> float:
    """Sine ease-in-out."""
    return -(math.cos(math.pi * t) - 1) / 2


def ease_in_expo(t: float) -> float:
    """Exponential ease-in."""
    return 0 if t == 0 else math.pow(2, 10 * (t - 1))


def ease_out_expo(t: float) -> float:
    """Exponential ease-out."""
    return 1 if t == 1 else 1 - math.pow(2, -10 * t)


def ease_in_out_expo(t: float) -> float:
    """Exponential ease-in-out."""
    if t == 0 or t == 1:
        return t

    if t < 0.5:
        return math.pow(2, 20 * t - 10) / 2
    else:
        return (2 - math.pow(2, -20 * t + 10)) / 2


def ease_in_circ(t: float) -> float:
    """Circular ease-in."""
    return 1 - math.sqrt(1 - t * t)


def ease_out_circ(t: float) -> float:
    """Circular ease-out."""
    t_shifted = t - 1
    return math.sqrt(1 - t_shifted * t_shifted)


def ease_in_out_circ(t: float) -> float:
    """Circular ease-in-out."""
    if t < 0.5:
        return (1 - math.sqrt(1 - 4 * t * t)) / 2
    else:
        t_shifted = -2 * t + 2
        return (math.sqrt(1 - t_shifted * t_shifted) + 1) / 2


# ==================== Easing Function Registry ====================

EASING_FUNCTIONS = {
    # Linear
    'linear': linear,

    # Quadratic
    'ease-in': ease_in,
    'ease-out': ease_out,
    'ease-in-out': ease_in_out,

    # Cubic
    'ease-in-cubic': ease_in_cubic,
    'ease-out-cubic': ease_out_cubic,
    'ease-in-out-cubic': ease_in_out_cubic,

    # Elastic
    'ease-in-elastic': ease_in_elastic,
    'ease-out-elastic': ease_out_elastic,
    'ease-in-out-elastic': ease_in_out_elastic,

    # Bounce
    'ease-in-bounce': ease_in_bounce,
    'ease-out-bounce': ease_out_bounce,
    'ease-in-out-bounce': ease_in_out_bounce,

    # Sine
    'ease-in-sine': ease_in_sine,
    'ease-out-sine': ease_out_sine,
    'ease-in-out-sine': ease_in_out_sine,

    # Exponential
    'ease-in-expo': ease_in_expo,
    'ease-out-expo': ease_out_expo,
    'ease-in-out-expo': ease_in_out_expo,

    # Circular
    'ease-in-circ': ease_in_circ,
    'ease-out-circ': ease_out_circ,
    'ease-in-out-circ': ease_in_out_circ,
}


def get_easing_function(name: str) -> EasingFunction:
    """
    Get easing function by name.

    Args:
        name: Easing function name

    Returns:
        Easing function

    Raises:
        ValueError: If easing function not found

    Example:
        >>> ease_func = get_easing_function('ease-in-out')
        >>> ease_func(0.5)
        0.5
    """
    if name not in EASING_FUNCTIONS:
        available = ', '.join(EASING_FUNCTIONS.keys())
        raise ValueError(f"Unknown easing function '{name}'. Available: {available}")

    return EASING_FUNCTIONS[name]
