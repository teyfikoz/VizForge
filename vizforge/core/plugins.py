"""
Plugin System for VizForge

Extensibility framework that allows custom chart types, data sources, and renderers.
Plotly limitation: Closed architecture, hard to extend.
VizForge solution: Open plugin system like VS Code.

Features:
- Custom chart types
- Custom data connectors
- Custom renderers
- Custom interactions
- Hot-reload support
- Plugin marketplace ready
"""

from typing import Dict, List, Optional, Callable, Any, Type
from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import json


@dataclass
class PluginMetadata:
    """Plugin metadata and information."""
    name: str
    version: str
    author: str
    description: str
    category: str  # 'chart', 'connector', 'renderer', 'interaction'
    dependencies: List[str]
    homepage: Optional[str] = None
    license: str = "MIT"


class Plugin:
    """
    Base class for all VizForge plugins.

    Create custom plugins by inheriting from this class.
    """

    metadata: PluginMetadata

    def __init__(self):
        """Initialize plugin."""
        if not hasattr(self, 'metadata'):
            raise NotImplementedError("Plugin must define metadata")

    def activate(self):
        """Called when plugin is activated."""
        pass

    def deactivate(self):
        """Called when plugin is deactivated."""
        pass

    def configure(self, config: Dict[str, Any]):
        """
        Configure plugin with user settings.

        Args:
            config: Configuration dictionary
        """
        pass


class ChartPlugin(Plugin):
    """
    Base class for custom chart type plugins.

    Example:
        class RadarChartPlugin(ChartPlugin):
            metadata = PluginMetadata(
                name='radar-chart',
                version='1.0.0',
                author='Your Name',
                description='Custom radar chart',
                category='chart'
            )

            def render(self, data, **kwargs):
                # Custom rendering logic
                pass
    """

    def render(self, data: Any, **kwargs) -> Any:
        """
        Render the custom chart.

        Args:
            data: Chart data
            **kwargs: Chart configuration

        Returns:
            Rendered chart object
        """
        raise NotImplementedError("Chart plugin must implement render()")

    def validate_data(self, data: Any) -> bool:
        """
        Validate input data for this chart type.

        Args:
            data: Input data

        Returns:
            True if data is valid
        """
        return True


class ConnectorPlugin(Plugin):
    """
    Base class for custom data connector plugins.

    Example:
        class SnowflakeConnector(ConnectorPlugin):
            metadata = PluginMetadata(
                name='snowflake',
                version='1.0.0',
                author='Your Name',
                description='Snowflake data connector',
                category='connector',
                dependencies=['snowflake-connector-python']
            )

            def connect(self, **credentials):
                # Connection logic
                pass

            def query(self, sql):
                # Query logic
                pass
    """

    def connect(self, **credentials) -> bool:
        """
        Establish connection to data source.

        Args:
            **credentials: Connection credentials

        Returns:
            True if connection successful
        """
        raise NotImplementedError("Connector plugin must implement connect()")

    def query(self, query: str) -> Any:
        """
        Execute query on data source.

        Args:
            query: Query string

        Returns:
            Query results
        """
        raise NotImplementedError("Connector plugin must implement query()")

    def disconnect(self):
        """Close connection to data source."""
        pass


class RendererPlugin(Plugin):
    """
    Base class for custom renderer plugins.

    Example:
        class SVGRenderer(RendererPlugin):
            metadata = PluginMetadata(
                name='svg-renderer',
                version='1.0.0',
                author='Your Name',
                description='SVG renderer',
                category='renderer'
            )

            def render(self, figure, format='svg'):
                # Rendering logic
                pass
    """

    def render(self, figure: Any, **options) -> Any:
        """
        Render figure to specific format.

        Args:
            figure: Figure object
            **options: Rendering options

        Returns:
            Rendered output
        """
        raise NotImplementedError("Renderer plugin must implement render()")


class InteractionPlugin(Plugin):
    """
    Base class for custom interaction plugins.

    Example:
        class VoiceControlPlugin(InteractionPlugin):
            metadata = PluginMetadata(
                name='voice-control',
                version='1.0.0',
                author='Your Name',
                description='Voice-controlled interactions',
                category='interaction'
            )

            def on_voice_command(self, command):
                # Handle voice command
                pass
    """

    def handle_event(self, event: str, data: Any):
        """
        Handle interaction event.

        Args:
            event: Event type
            data: Event data
        """
        raise NotImplementedError("Interaction plugin must implement handle_event()")


class PluginManager:
    """
    Manages VizForge plugins.

    Unlike Plotly's closed system, VizForge is fully extensible.
    """

    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_dirs: List[Path] = [
            Path.home() / '.vizforge' / 'plugins',
            Path(__file__).parent.parent / 'plugins'
        ]
        self._hooks: Dict[str, List[Callable]] = {}

    def discover_plugins(self):
        """
        Discover and load plugins from plugin directories.

        Scans plugin directories for valid plugin modules.
        """
        discovered = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue

            # Scan for Python files
            for plugin_file in plugin_dir.glob('*.py'):
                if plugin_file.name.startswith('_'):
                    continue

                try:
                    plugin = self._load_plugin_file(plugin_file)
                    if plugin:
                        discovered.append(plugin)
                except Exception as e:
                    print(f"Failed to load plugin {plugin_file}: {e}")

        print(f"Discovered {len(discovered)} plugins")
        return discovered

    def _load_plugin_file(self, filepath: Path) -> Optional[Plugin]:
        """Load plugin from Python file."""
        # Import the module
        spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find Plugin subclass
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Plugin) and obj != Plugin:
                # Instantiate plugin
                plugin = obj()
                return plugin

        return None

    def register(self, plugin: Plugin):
        """
        Register a plugin.

        Args:
            plugin: Plugin instance
        """
        plugin_name = plugin.metadata.name

        if plugin_name in self.plugins:
            raise ValueError(f"Plugin '{plugin_name}' already registered")

        self.plugins[plugin_name] = plugin
        plugin.activate()

        print(f"Registered plugin: {plugin_name} v{plugin.metadata.version}")

    def unregister(self, plugin_name: str):
        """
        Unregister a plugin.

        Args:
            plugin_name: Name of plugin to unregister
        """
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")

        plugin = self.plugins[plugin_name]
        plugin.deactivate()
        del self.plugins[plugin_name]

        print(f"Unregistered plugin: {plugin_name}")

    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """
        Get registered plugin by name.

        Args:
            plugin_name: Plugin name

        Returns:
            Plugin instance or None
        """
        return self.plugins.get(plugin_name)

    def list_plugins(self, category: Optional[str] = None) -> List[PluginMetadata]:
        """
        List all registered plugins.

        Args:
            category: Filter by category (optional)

        Returns:
            List of plugin metadata
        """
        plugins = []

        for plugin in self.plugins.values():
            if category is None or plugin.metadata.category == category:
                plugins.append(plugin.metadata)

        return plugins

    def add_hook(self, hook_name: str, callback: Callable):
        """
        Add callback hook for plugin events.

        Args:
            hook_name: Name of hook
            callback: Callback function
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []

        self._hooks[hook_name].append(callback)

    def trigger_hook(self, hook_name: str, *args, **kwargs):
        """
        Trigger all callbacks for a hook.

        Args:
            hook_name: Name of hook
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks
        """
        if hook_name not in self._hooks:
            return

        for callback in self._hooks[hook_name]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Hook callback error: {e}")

    def create_plugin_template(self, name: str, category: str, output_dir: Path):
        """
        Create a plugin template for development.

        Args:
            name: Plugin name
            category: Plugin category
            output_dir: Output directory
        """
        template = f'''"""
{name.title()} Plugin for VizForge

Auto-generated plugin template.
"""

from vizforge.core.plugins import {category.title()}Plugin, PluginMetadata


class {name.title().replace("-", "")}Plugin({category.title()}Plugin):
    """Custom {name} plugin."""

    metadata = PluginMetadata(
        name='{name}',
        version='1.0.0',
        author='Your Name',
        description='{name} plugin for VizForge',
        category='{category}',
        dependencies=[]
    )

    def activate(self):
        """Activate plugin."""
        print(f"{{self.metadata.name}} activated")

    def deactivate(self):
        """Deactivate plugin."""
        print(f"{{self.metadata.name}} deactivated")

    # Implement required methods based on plugin type
'''

        output_file = output_dir / f"{name}.py"
        output_file.write_text(template)

        print(f"Created plugin template: {output_file}")


# Global plugin manager
_plugin_manager = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


# Convenience functions
def register_plugin(plugin: Plugin):
    """Register a plugin with the global plugin manager."""
    manager = get_plugin_manager()
    manager.register(plugin)


def get_plugin(name: str) -> Optional[Plugin]:
    """Get a registered plugin by name."""
    manager = get_plugin_manager()
    return manager.get_plugin(name)


def list_plugins(category: Optional[str] = None) -> List[PluginMetadata]:
    """List all registered plugins."""
    manager = get_plugin_manager()
    return manager.list_plugins(category)
