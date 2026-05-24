"""Basic tests for VizForge package."""

import pytest


def test_import():
    """Test that vizforge can be imported."""
    import vizforge
    assert hasattr(vizforge, "__version__")


def test_version():
    """Test version string format."""
    from vizforge.version import __version__, __version_info__
    assert isinstance(__version__, str)
    assert len(__version__.split(".")) == 3
    assert __version_info__ == (3, 3, 0)


def test_config_import():
    """Test config module import."""
    from vizforge import config
    assert config is not None


def test_charts_module():
    """Test charts module is importable."""
    from vizforge import charts
    assert charts is not None


def test_utils_module():
    """Test utils module is importable."""
    from vizforge import utils
    assert utils is not None


def test_version_feature_flags():
    """Test that feature flags are defined."""
    from vizforge.version import CHARTS_2D_ENABLED, WEBGPU_RENDERING
    assert CHARTS_2D_ENABLED is True
    assert WEBGPU_RENDERING is True
