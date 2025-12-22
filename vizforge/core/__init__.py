"""Core VizForge functionality."""

from .base import BaseChart
from .theme import Theme, get_theme, set_theme, register_theme, list_themes

# VizForge v1.0.0 NEW: Performance & Rendering
from .engine import RenderingEngine, AnimationEngine, AnimationConfig
from .cache import ChartCache, get_global_cache, clear_cache
from .accessibility import AccessibilityHelper, AccessibilityLevel, ColorBlindMode

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
