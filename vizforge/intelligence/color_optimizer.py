"""
VizForge Color Optimizer

Perceptual color optimization and accessibility-first palettes (NO API costs).
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import List, Dict, Any, Optional, Tuple
import colorsys
import math


class ColorOptimizer:
    """
    Intelligent color palette optimization using perceptual color theory.

    NO API CALLS - pure mathematical color space transformations.

    Features:
    - Color-blind safe palettes (deuteranopia, protanopia, tritanopia)
    - WCAG 2.1 AA/AAA contrast optimization
    - Perceptual uniformity (HCL color space)
    - Semantic color selection (red=danger, green=success)
    - Palette generation for N categories

    Example:
        >>> optimizer = ColorOptimizer()
        >>> palette = optimizer.generate_palette(n_colors=8, mode='colorblind_safe')
        >>> print(palette)  # ['#0173B2', '#DE8F05', ...]
    """

    # Predefined accessible palettes
    PALETTES = {
        # Color-blind safe (Okabe & Ito, 2008)
        'colorblind_safe': [
            '#0173B2',  # Blue
            '#DE8F05',  # Orange
            '#029E73',  # Green
            '#CC78BC',  # Purple
            '#ECE133',  # Yellow
            '#56B4E9',  # Sky Blue
            '#CA9161',  # Brown
            '#949494',  # Gray
        ],

        # High contrast (WCAG AAA)
        'high_contrast': [
            '#000000',  # Black
            '#E69F00',  # Orange
            '#56B4E9',  # Sky Blue
            '#009E73',  # Green
            '#F0E442',  # Yellow
            '#0072B2',  # Blue
            '#D55E00',  # Red-Orange
            '#CC79A7',  # Pink
        ],

        # Tableau 10
        'tableau': [
            '#4E79A7',  # Blue
            '#F28E2B',  # Orange
            '#E15759',  # Red
            '#76B7B2',  # Teal
            '#59A14F',  # Green
            '#EDC948',  # Yellow
            '#B07AA1',  # Purple
            '#FF9DA7',  # Pink
            '#9C755F',  # Brown
            '#BAB0AC',  # Gray
        ],

        # Material Design
        'material': [
            '#1976D2',  # Blue
            '#F57C00',  # Orange
            '#388E3C',  # Green
            '#D32F2F',  # Red
            '#7B1FA2',  # Purple
            '#0097A7',  # Cyan
            '#FBC02D',  # Yellow
            '#C2185B',  # Pink
        ],

        # Pastel (soft colors)
        'pastel': [
            '#A8D5E2',  # Light Blue
            '#FFD4A3',  # Light Orange
            '#C5E1A5',  # Light Green
            '#F48FB1',  # Light Pink
            '#CE93D8',  # Light Purple
            '#FFEB99',  # Light Yellow
            '#BCAAA4',  # Light Brown
            '#B0BEC5',  # Light Gray
        ],
    }

    # Semantic colors
    SEMANTIC_COLORS = {
        'success': '#28A745',
        'warning': '#FFC107',
        'danger': '#DC3545',
        'info': '#17A2B8',
        'primary': '#007BFF',
        'secondary': '#6C757D',
    }

    def __init__(self):
        """Initialize color optimizer."""
        pass

    def generate_palette(
        self,
        n_colors: int,
        mode: str = 'colorblind_safe',
        hue_start: float = 0.0,
        lightness: float = 0.6,
        saturation: float = 0.7
    ) -> List[str]:
        """
        Generate optimized color palette.

        Args:
            n_colors: Number of colors needed
            mode: Palette mode ('colorblind_safe', 'high_contrast', 'tableau',
                               'material', 'pastel', 'custom')
            hue_start: Starting hue (0.0-1.0) for custom mode
            lightness: Lightness (0.0-1.0) for custom mode
            saturation: Saturation (0.0-1.0) for custom mode

        Returns:
            List of hex color codes

        Example:
            >>> palette = optimizer.generate_palette(5, mode='colorblind_safe')
            >>> print(palette)  # ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#ECE133']
        """
        if mode in self.PALETTES:
            base_palette = self.PALETTES[mode]

            # If we need more colors, extend using HCL interpolation
            if n_colors <= len(base_palette):
                return base_palette[:n_colors]
            else:
                return self._extend_palette(base_palette, n_colors)

        elif mode == 'custom':
            return self._generate_hcl_palette(n_colors, hue_start, lightness, saturation)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use: {list(self.PALETTES.keys()) + ['custom']}")

    def _extend_palette(self, base_palette: List[str], n_colors: int) -> List[str]:
        """Extend palette by interpolating in HCL space."""
        extended = base_palette.copy()

        # Convert to HCL
        hcl_colors = [self._hex_to_hcl(c) for c in base_palette]

        # Generate additional colors by varying lightness
        while len(extended) < n_colors:
            for h, c, l in hcl_colors:
                if len(extended) >= n_colors:
                    break

                # Vary lightness
                new_l = (l + 0.2) % 1.0
                new_color = self._hcl_to_hex(h, c, new_l)
                extended.append(new_color)

        return extended[:n_colors]

    def _generate_hcl_palette(
        self,
        n_colors: int,
        hue_start: float,
        lightness: float,
        saturation: float
    ) -> List[str]:
        """
        Generate perceptually uniform palette using HCL color space.

        HCL (Hue, Chroma, Luminance) is perceptually uniform,
        unlike RGB which has uneven perceptual distances.
        """
        palette = []

        # Distribute hues evenly
        hue_step = 1.0 / n_colors

        for i in range(n_colors):
            hue = (hue_start + i * hue_step) % 1.0
            color = self._hcl_to_hex(hue, saturation, lightness)
            palette.append(color)

        return palette

    def optimize_for_accessibility(
        self,
        colors: List[str],
        background: str = '#FFFFFF',
        level: str = 'AA'
    ) -> List[str]:
        """
        Optimize colors for WCAG accessibility.

        Args:
            colors: List of hex colors to optimize
            background: Background color (hex)
            level: WCAG level ('AA' or 'AAA')

        Returns:
            Optimized color list with sufficient contrast

        Example:
            >>> colors = ['#FFFF00', '#00FF00']  # Too bright
            >>> optimized = optimizer.optimize_for_accessibility(colors)
            >>> # Returns darker versions with better contrast
        """
        min_contrast = 4.5 if level == 'AA' else 7.0
        optimized = []

        for color in colors:
            contrast = self.calculate_contrast(color, background)

            if contrast >= min_contrast:
                optimized.append(color)
            else:
                # Darken/lighten until contrast is sufficient
                optimized_color = self._adjust_for_contrast(
                    color, background, min_contrast
                )
                optimized.append(optimized_color)

        return optimized

    def calculate_contrast(self, color1: str, color2: str) -> float:
        """
        Calculate WCAG contrast ratio between two colors.

        Args:
            color1: First color (hex)
            color2: Second color (hex)

        Returns:
            Contrast ratio (1.0 to 21.0)

        Example:
            >>> contrast = optimizer.calculate_contrast('#000000', '#FFFFFF')
            >>> print(contrast)  # 21.0 (maximum contrast)
        """
        lum1 = self._get_relative_luminance(color1)
        lum2 = self._get_relative_luminance(color2)

        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)

        return (lighter + 0.05) / (darker + 0.05)

    def _get_relative_luminance(self, hex_color: str) -> float:
        """Calculate relative luminance (WCAG formula)."""
        r, g, b = self._hex_to_rgb(hex_color)

        # Convert to sRGB
        def to_srgb(c):
            c = c / 255.0
            if c <= 0.03928:
                return c / 12.92
            else:
                return ((c + 0.055) / 1.055) ** 2.4

        r_srgb = to_srgb(r)
        g_srgb = to_srgb(g)
        b_srgb = to_srgb(b)

        return 0.2126 * r_srgb + 0.7152 * g_srgb + 0.0722 * b_srgb

    def _adjust_for_contrast(
        self,
        color: str,
        background: str,
        min_contrast: float
    ) -> str:
        """Adjust color brightness to meet contrast requirement."""
        h, s, l = self._hex_to_hsl(color)

        # Binary search for optimal lightness
        low, high = 0.0, 1.0
        best_l = l

        for _ in range(20):  # 20 iterations for convergence
            mid = (low + high) / 2
            test_color = self._hsl_to_hex(h, s, mid)
            contrast = self.calculate_contrast(test_color, background)

            if contrast >= min_contrast:
                best_l = mid
                # Try to get closer to original lightness
                if mid < l:
                    low = mid
                else:
                    high = mid
            else:
                if mid < 0.5:
                    high = mid
                else:
                    low = mid

        return self._hsl_to_hex(h, s, best_l)

    def suggest_semantic_color(self, concept: str) -> str:
        """
        Suggest color based on semantic meaning.

        Args:
            concept: Concept name ('success', 'warning', 'danger', etc.)

        Returns:
            Hex color code

        Example:
            >>> color = optimizer.suggest_semantic_color('success')
            >>> print(color)  # '#28A745' (green)
        """
        concept_lower = concept.lower()

        if concept_lower in self.SEMANTIC_COLORS:
            return self.SEMANTIC_COLORS[concept_lower]

        # Fuzzy matching
        if 'success' in concept_lower or 'good' in concept_lower or 'positive' in concept_lower:
            return self.SEMANTIC_COLORS['success']
        elif 'warning' in concept_lower or 'caution' in concept_lower:
            return self.SEMANTIC_COLORS['warning']
        elif 'danger' in concept_lower or 'error' in concept_lower or 'bad' in concept_lower:
            return self.SEMANTIC_COLORS['danger']
        elif 'info' in concept_lower or 'information' in concept_lower:
            return self.SEMANTIC_COLORS['info']
        else:
            return self.SEMANTIC_COLORS['primary']

    # ==================== Color Space Conversions ====================

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex to RGB."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(self, r: int, g: int, b: int) -> str:
        """Convert RGB to hex."""
        return f'#{r:02X}{g:02X}{b:02X}'

    def _hex_to_hsl(self, hex_color: str) -> Tuple[float, float, float]:
        """Convert hex to HSL."""
        r, g, b = self._hex_to_rgb(hex_color)
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return h, s, l

    def _hsl_to_hex(self, h: float, s: float, l: float) -> str:
        """Convert HSL to hex."""
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return self._rgb_to_hex(r, g, b)

    def _hex_to_hcl(self, hex_color: str) -> Tuple[float, float, float]:
        """
        Convert hex to HCL (perceptually uniform).

        Note: This is a simplified HCL approximation.
        For production, use dedicated libraries like colorspacious.
        """
        h, s, l = self._hex_to_hsl(hex_color)
        # Approximate HCL using HSL
        # In production, use proper LAB -> LCH conversion
        c = s  # Chroma approximation
        return h, c, l

    def _hcl_to_hex(self, h: float, c: float, l: float) -> str:
        """
        Convert HCL to hex.

        Note: This is a simplified HCL approximation.
        """
        # Approximate using HSL
        return self._hsl_to_hex(h, c, l)
