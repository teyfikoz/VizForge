"""Core VizForge functionality."""

from .accessibility import AccessibilityHelper, AccessibilityLevel, ColorBlindMode
from .base import BaseChart
from .cache import ChartCache, clear_cache, get_global_cache

# VizForge v1.0.0 NEW: Performance & Rendering
from .engine import AnimationConfig, AnimationEngine, RenderingEngine
from .theme import Theme, get_theme, list_themes, register_theme, set_theme

__all__ = [
    # Core v0.5.x (backward compatible)
    "BaseChart",
    "Theme",
    "get_theme",
    "set_theme",
    "register_theme",
    "list_themes",

    # NEW v1.0.0: Rendering & Performance
    "RenderingEngine",
    "AnimationEngine",
    "AnimationConfig",
    "ChartCache",
    "get_global_cache",
    "clear_cache",
    "AccessibilityHelper",
    "AccessibilityLevel",
    "ColorBlindMode",
]
