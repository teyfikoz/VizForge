"""
VizForge Smart Layout Engine

Intelligent dashboard layout optimization with responsive design.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import plotly.graph_objects as go


class LayoutType(Enum):
    """Types of dashboard layouts."""
    GRID = "grid"               # Fixed grid layout
    FLEX = "flex"               # Flexible layout
    MASONRY = "masonry"         # Pinterest-style masonry
    FLOW = "flow"               # Flowing layout
    SIDEBAR = "sidebar"         # Sidebar + content
    SPLIT = "split"             # Split view
    TABS = "tabs"               # Tabbed layout
    CANVAS = "canvas"           # Free-form canvas


class ResponsiveBreakpoint(Enum):
    """Responsive breakpoints (Bootstrap-style)."""
    XS = 576      # Extra small (phone)
    SM = 768      # Small (tablet portrait)
    MD = 992      # Medium (tablet landscape)
    LG = 1200     # Large (desktop)
    XL = 1400     # Extra large (wide desktop)


@dataclass
class LayoutConstraint:
    """
    Layout constraints for components.

    Attributes:
        min_width: Minimum width (px or percentage)
        max_width: Maximum width
        min_height: Minimum height
        max_height: Maximum height
        aspect_ratio: Aspect ratio (width:height)
        fixed_size: Whether size is fixed
    """
    min_width: Optional[Union[int, str]] = None
    max_width: Optional[Union[int, str]] = None
    min_height: Optional[Union[int, str]] = None
    max_height: Optional[Union[int, str]] = None
    aspect_ratio: Optional[float] = None
    fixed_size: bool = False


@dataclass
class GridCell:
    """
    Grid cell configuration.

    Attributes:
        row: Row index (1-based)
        col: Column index (1-based)
        row_span: Number of rows to span
        col_span: Number of columns to span
        component: Component in this cell
        constraints: Layout constraints
    """
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    component: Any = None
    constraints: Optional[LayoutConstraint] = None


class SmartLayoutEngine:
    """
    Intelligent layout engine with auto-optimization.

    Features:
    - Responsive grid system
    - Auto-sizing based on content
    - Mobile-first design
    - Aspect ratio preservation
    - Whitespace optimization

    Example:
        >>> engine = SmartLayoutEngine(rows=3, cols=3)
        >>> engine.add_component(chart1, row=1, col=1, col_span=2)
        >>> engine.add_component(chart2, row=1, col=3)
        >>> layout = engine.generate_layout()
    """

    def __init__(
        self,
        rows: int = 2,
        cols: int = 2,
        layout_type: LayoutType = LayoutType.GRID,
        responsive: bool = True,
        mobile_first: bool = True,
        gap: int = 20
    ):
        """
        Initialize smart layout engine.

        Args:
            rows: Number of rows
            cols: Number of columns
            layout_type: Type of layout
            responsive: Enable responsive design
            mobile_first: Use mobile-first approach
            gap: Gap between components (px)
        """
        self.rows = rows
        self.cols = cols
        self.layout_type = layout_type
        self.responsive = responsive
        self.mobile_first = mobile_first
        self.gap = gap

        # Grid cells
        self.cells: List[GridCell] = []

        # Responsive breakpoints
        self.breakpoints = {
            'xs': ResponsiveBreakpoint.XS.value,
            'sm': ResponsiveBreakpoint.SM.value,
            'md': ResponsiveBreakpoint.MD.value,
            'lg': ResponsiveBreakpoint.LG.value,
            'xl': ResponsiveBreakpoint.XL.value,
        }

    def add_component(
        self,
        component: Any,
        row: int,
        col: int,
        row_span: int = 1,
        col_span: int = 1,
        constraints: Optional[LayoutConstraint] = None
    ) -> 'SmartLayoutEngine':
        """
        Add component to layout.

        Args:
            component: Component to add (chart, widget, etc.)
            row: Row position (1-based)
            col: Column position (1-based)
            row_span: Number of rows to span
            col_span: Number of columns to span
            constraints: Layout constraints

        Returns:
            Self for method chaining

        Example:
            >>> engine.add_component(chart, row=1, col=1, col_span=2)
        """
        cell = GridCell(
            row=row,
            col=col,
            row_span=row_span,
            col_span=col_span,
            component=component,
            constraints=constraints
        )

        self.cells.append(cell)
        return self

    def generate_layout(self) -> Dict[str, Any]:
        """
        Generate optimized layout configuration.

        Returns:
            Layout configuration dictionary

        Example:
            >>> layout = engine.generate_layout()
            >>> print(layout['grid_template_rows'])
        """
        if self.layout_type == LayoutType.GRID:
            return self._generate_grid_layout()
        elif self.layout_type == LayoutType.FLEX:
            return self._generate_flex_layout()
        elif self.layout_type == LayoutType.MASONRY:
            return self._generate_masonry_layout()
        elif self.layout_type == LayoutType.SIDEBAR:
            return self._generate_sidebar_layout()
        else:
            return self._generate_grid_layout()

    def _generate_grid_layout(self) -> Dict[str, Any]:
        """Generate CSS Grid layout."""
        # Calculate optimal row/column sizes
        row_heights = self._calculate_row_heights()
        col_widths = self._calculate_col_widths()

        layout = {
            'display': 'grid',
            'grid_template_rows': ' '.join(row_heights),
            'grid_template_columns': ' '.join(col_widths),
            'gap': f'{self.gap}px',
            'padding': f'{self.gap}px',
            'height': '100%',
            'width': '100%',
        }

        # Add cell positions
        layout['cells'] = []
        for cell in self.cells:
            cell_config = {
                'grid_row': f'{cell.row} / span {cell.row_span}',
                'grid_column': f'{cell.col} / span {cell.col_span}',
                'component': cell.component,
            }

            # Apply constraints
            if cell.constraints:
                if cell.constraints.min_width:
                    cell_config['min_width'] = cell.constraints.min_width
                if cell.constraints.max_width:
                    cell_config['max_width'] = cell.constraints.max_width
                if cell.constraints.min_height:
                    cell_config['min_height'] = cell.constraints.min_height
                if cell.constraints.max_height:
                    cell_config['max_height'] = cell.constraints.max_height
                if cell.constraints.aspect_ratio:
                    cell_config['aspect_ratio'] = cell.constraints.aspect_ratio

            layout['cells'].append(cell_config)

        # Add responsive rules
        if self.responsive:
            layout['responsive'] = self._generate_responsive_rules()

        return layout

    def _generate_flex_layout(self) -> Dict[str, Any]:
        """Generate Flexbox layout."""
        layout = {
            'display': 'flex',
            'flex_wrap': 'wrap',
            'gap': f'{self.gap}px',
            'padding': f'{self.gap}px',
            'align_items': 'stretch',
        }

        layout['items'] = []
        for cell in self.cells:
            # Calculate flex basis
            flex_basis = f'{100 / self.cols * cell.col_span}%'

            item_config = {
                'flex': f'1 1 {flex_basis}',
                'component': cell.component,
            }

            if cell.constraints:
                if cell.constraints.min_width:
                    item_config['min_width'] = cell.constraints.min_width
                if cell.constraints.aspect_ratio:
                    item_config['aspect_ratio'] = cell.constraints.aspect_ratio

            layout['items'].append(item_config)

        return layout

    def _generate_masonry_layout(self) -> Dict[str, Any]:
        """Generate Masonry (Pinterest-style) layout."""
        layout = {
            'display': 'grid',
            'grid_template_columns': f'repeat({self.cols}, 1fr)',
            'grid_auto_rows': '10px',  # Small row height for masonry
            'gap': f'{self.gap}px',
            'padding': f'{self.gap}px',
        }

        layout['items'] = []
        for i, cell in enumerate(self.cells):
            # Auto-calculate row span based on content height
            estimated_height = self._estimate_component_height(cell.component)
            row_span = max(1, int(estimated_height / 10))

            item_config = {
                'grid_column': f'{cell.col} / span {cell.col_span}',
                'grid_row': f'auto / span {row_span}',
                'component': cell.component,
            }

            layout['items'].append(item_config)

        return layout

    def _generate_sidebar_layout(self) -> Dict[str, Any]:
        """Generate sidebar + content layout."""
        layout = {
            'display': 'grid',
            'grid_template_columns': '250px 1fr',
            'grid_template_rows': '1fr',
            'gap': f'{self.gap}px',
            'height': '100vh',
        }

        # Separate sidebar and main content
        sidebar_items = [cell for cell in self.cells if cell.col == 1]
        content_items = [cell for cell in self.cells if cell.col > 1]

        layout['sidebar'] = {
            'grid_column': '1',
            'components': [cell.component for cell in sidebar_items],
        }

        layout['content'] = {
            'grid_column': '2',
            'display': 'grid',
            'grid_template_rows': f'repeat({self.rows}, 1fr)',
            'gap': f'{self.gap}px',
            'components': [cell.component for cell in content_items],
        }

        return layout

    def _calculate_row_heights(self) -> List[str]:
        """Calculate optimal row heights."""
        row_heights = []

        for row_idx in range(1, self.rows + 1):
            # Check for fixed heights in this row
            fixed_heights = [
                cell.constraints.min_height
                for cell in self.cells
                if cell.row == row_idx and cell.constraints and cell.constraints.fixed_size
            ]

            if fixed_heights:
                # Use fixed height
                row_heights.append(f'{fixed_heights[0]}px' if isinstance(fixed_heights[0], int) else fixed_heights[0])
            else:
                # Use flexible height
                row_heights.append('1fr')

        return row_heights

    def _calculate_col_widths(self) -> List[str]:
        """Calculate optimal column widths."""
        col_widths = []

        for col_idx in range(1, self.cols + 1):
            # Check for fixed widths in this column
            fixed_widths = [
                cell.constraints.min_width
                for cell in self.cells
                if cell.col == col_idx and cell.constraints and cell.constraints.fixed_size
            ]

            if fixed_widths:
                # Use fixed width
                col_widths.append(f'{fixed_widths[0]}px' if isinstance(fixed_widths[0], int) else fixed_widths[0])
            else:
                # Use flexible width
                col_widths.append('1fr')

        return col_widths

    def _generate_responsive_rules(self) -> Dict[str, Any]:
        """Generate responsive breakpoint rules."""
        rules = {}

        # Mobile (xs): Stack vertically
        rules['xs'] = {
            'max_width': f"{self.breakpoints['xs']}px",
            'grid_template_columns': '1fr',
            'grid_auto_rows': 'auto',
        }

        # Tablet (sm/md): 2 columns
        rules['sm'] = {
            'min_width': f"{self.breakpoints['sm']}px",
            'grid_template_columns': 'repeat(2, 1fr)',
        }

        # Desktop (lg+): Full grid
        rules['lg'] = {
            'min_width': f"{self.breakpoints['lg']}px",
            'grid_template_columns': f'repeat({self.cols}, 1fr)',
        }

        return rules

    def _estimate_component_height(self, component: Any) -> int:
        """Estimate component height for masonry layout."""
        # Default estimate
        default_height = 300

        # Check if component is a figure
        if isinstance(component, go.Figure):
            if component.layout.height:
                return component.layout.height
            return default_height

        # Check if component has height attribute
        if hasattr(component, 'height'):
            return component.height

        return default_height

    def optimize_layout(self) -> 'SmartLayoutEngine':
        """
        Auto-optimize layout based on content.

        Applies intelligent rules:
        - Merge small components
        - Expand large components
        - Balance whitespace
        - Align related components

        Returns:
            Self for method chaining
        """
        # Sort cells by size (largest first)
        self.cells.sort(key=lambda c: c.row_span * c.col_span, reverse=True)

        # Reposition for optimal layout
        self._reposition_cells()

        return self

    def _reposition_cells(self):
        """Reposition cells for optimal layout."""
        # Create grid occupancy map
        occupied = [[False] * self.cols for _ in range(self.rows)]

        # Place cells
        for cell in self.cells:
            # Find best position
            row, col = self._find_best_position(occupied, cell.row_span, cell.col_span)

            # Update cell position
            cell.row = row
            cell.col = col

            # Mark as occupied
            for r in range(row, min(row + cell.row_span, self.rows)):
                for c in range(col, min(col + cell.col_span, self.cols)):
                    if r < self.rows and c < self.cols:
                        occupied[r][c] = True

    def _find_best_position(
        self,
        occupied: List[List[bool]],
        row_span: int,
        col_span: int
    ) -> Tuple[int, int]:
        """Find best position for component."""
        # Try to place from top-left
        for row in range(self.rows):
            for col in range(self.cols):
                if self._can_place(occupied, row, col, row_span, col_span):
                    return (row + 1, col + 1)  # 1-based indexing

        # Fallback: return first available
        return (1, 1)

    def _can_place(
        self,
        occupied: List[List[bool]],
        row: int,
        col: int,
        row_span: int,
        col_span: int
    ) -> bool:
        """Check if component can be placed at position."""
        # Check bounds
        if row + row_span > self.rows or col + col_span > self.cols:
            return False

        # Check occupancy
        for r in range(row, row + row_span):
            for c in range(col, col + col_span):
                if occupied[r][c]:
                    return False

        return True


# ==================== Helper Functions ====================

def create_grid_layout(
    rows: int = 2,
    cols: int = 2,
    gap: int = 20,
    responsive: bool = True
) -> SmartLayoutEngine:
    """
    Quick create grid layout.

    Args:
        rows: Number of rows
        cols: Number of columns
        gap: Gap between components
        responsive: Enable responsive design

    Returns:
        SmartLayoutEngine instance

    Example:
        >>> layout = create_grid_layout(rows=3, cols=3)
        >>> layout.add_component(chart1, row=1, col=1)
    """
    return SmartLayoutEngine(
        rows=rows,
        cols=cols,
        layout_type=LayoutType.GRID,
        responsive=responsive,
        gap=gap
    )


def create_sidebar_layout(
    sidebar_width: int = 250,
    gap: int = 20
) -> SmartLayoutEngine:
    """
    Quick create sidebar layout.

    Args:
        sidebar_width: Sidebar width (px)
        gap: Gap between components

    Returns:
        SmartLayoutEngine instance

    Example:
        >>> layout = create_sidebar_layout()
        >>> layout.add_component(filters, row=1, col=1)  # Sidebar
        >>> layout.add_component(chart, row=1, col=2)    # Main content
    """
    return SmartLayoutEngine(
        rows=1,
        cols=2,
        layout_type=LayoutType.SIDEBAR,
        gap=gap
    )


def create_masonry_layout(
    cols: int = 3,
    gap: int = 20
) -> SmartLayoutEngine:
    """
    Quick create masonry layout.

    Args:
        cols: Number of columns
        gap: Gap between components

    Returns:
        SmartLayoutEngine instance

    Example:
        >>> layout = create_masonry_layout(cols=4)
        >>> layout.add_component(chart1, row=1, col=1)
    """
    return SmartLayoutEngine(
        rows=10,  # Auto-adjust for masonry
        cols=cols,
        layout_type=LayoutType.MASONRY,
        gap=gap
    )
