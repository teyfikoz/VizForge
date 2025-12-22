"""
VizForge Animations Module Tests

Tests for transitions, easing functions, and gestures.
Target: 90%+ coverage for animations module.
"""

import pytest
import math
import plotly.graph_objects as go

# Import animations module components
from ..animations.easing import (
    linear, ease_in, ease_out, ease_in_out,
    ease_in_cubic, ease_out_cubic, ease_in_out_cubic,
    ease_in_elastic, ease_out_elastic,
    ease_in_bounce, ease_out_bounce,
    bezier, get_easing_function
)
from ..animations.transitions import (
    TransitionConfig, TransitionType, AnimationEngine,
    apply_transition, create_transition
)
from ..animations.gestures import (
    GestureType, GestureConfig, GestureHandler, TouchEvent,
    setup_gestures, enable_mobile_gestures, optimize_for_touch
)


# ==================== Easing Function Tests ====================

class TestEasingFunctions:
    """Tests for easing functions."""

    def test_linear(self):
        """Test linear easing."""
        assert linear(0.0) == 0.0
        assert linear(0.5) == 0.5
        assert linear(1.0) == 1.0

    def test_ease_in(self):
        """Test ease-in (quadratic)."""
        assert ease_in(0.0) == 0.0
        assert ease_in(0.5) == 0.25
        assert ease_in(1.0) == 1.0

    def test_ease_out(self):
        """Test ease-out (quadratic)."""
        assert ease_out(0.0) == 0.0
        assert abs(ease_out(0.5) - 0.75) < 0.01
        assert ease_out(1.0) == 1.0

    def test_ease_in_out(self):
        """Test ease-in-out (quadratic)."""
        assert ease_in_out(0.0) == 0.0
        assert abs(ease_in_out(0.5) - 0.5) < 0.01
        assert ease_in_out(1.0) == 1.0

    def test_cubic_functions(self):
        """Test cubic easing functions."""
        # Cubic should be more pronounced than quadratic
        assert ease_in_cubic(0.5) < ease_in(0.5)
        assert ease_out_cubic(0.5) > ease_out(0.5)

    def test_elastic_endpoints(self):
        """Test elastic functions at endpoints."""
        assert ease_in_elastic(0.0) == 0.0
        assert ease_in_elastic(1.0) == 1.0
        assert ease_out_elastic(0.0) == 0.0
        assert ease_out_elastic(1.0) == 1.0

    def test_bounce_endpoints(self):
        """Test bounce functions at endpoints."""
        assert ease_in_bounce(0.0) == 0.0
        assert abs(ease_in_bounce(1.0) - 1.0) < 0.01
        assert ease_out_bounce(0.0) == 0.0
        assert abs(ease_out_bounce(1.0) - 1.0) < 0.01

    def test_bezier_linear(self):
        """Test bezier with linear control points."""
        # Linear bezier: P0=0, P1=0.33, P2=0.66, P3=1
        ease_func = bezier(0.0, 0.33, 0.66, 1.0)

        assert abs(ease_func(0.0) - 0.0) < 0.01
        assert abs(ease_func(1.0) - 1.0) < 0.01

    def test_get_easing_function_valid(self):
        """Test getting easing function by name."""
        func = get_easing_function('ease-in-out')
        assert func is not None
        assert callable(func)

    def test_get_easing_function_invalid(self):
        """Test getting invalid easing function."""
        with pytest.raises(ValueError):
            get_easing_function('invalid-easing')

    def test_all_easing_functions_range(self):
        """Test all easing functions return values in reasonable range."""
        functions = [
            linear, ease_in, ease_out, ease_in_out,
            ease_in_cubic, ease_out_cubic, ease_in_out_cubic,
            ease_in_bounce, ease_out_bounce,
        ]

        for func in functions:
            # Test at various points
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = func(t)
                # Most easing functions should stay roughly in 0-1 range
                # (elastic and bounce may overshoot slightly)
                assert result >= -0.5  # Allow some overshoot
                assert result <= 1.5   # Allow some overshoot


# ==================== TransitionConfig Tests ====================

class TestTransitionConfig:
    """Tests for TransitionConfig."""

    def test_default_config(self):
        """Test default transition configuration."""
        config = TransitionConfig()

        assert config.transition_type == TransitionType.SMOOTH
        assert config.duration == 500
        assert config.easing == 'ease-in-out'

    def test_custom_config(self):
        """Test custom transition configuration."""
        config = TransitionConfig(
            transition_type=TransitionType.ELASTIC,
            duration=1000,
            easing='ease-out-bounce'
        )

        assert config.transition_type == TransitionType.ELASTIC
        assert config.duration == 1000
        assert config.easing == 'ease-out-bounce'


# ==================== AnimationEngine Tests ====================

class TestAnimationEngine:
    """Tests for AnimationEngine."""

    @pytest.fixture
    def sample_figure(self):
        """Create sample Plotly figure."""
        return go.Figure(
            data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3])],
            layout=go.Layout(title='Test Chart')
        )

    def test_apply_transition(self, sample_figure):
        """Test applying transition to figure."""
        config = TransitionConfig(duration=800, easing='ease-in-out')

        result = AnimationEngine.apply_transition(sample_figure, config)

        assert result.layout.transition is not None
        assert result.layout.transition['duration'] == 800

    def test_create_frames(self):
        """Test creating animation frames."""
        states = [
            {'x': [1, 2, 3], 'y': [1, 2, 3]},
            {'x': [1, 2, 3], 'y': [2, 3, 4]},
            {'x': [1, 2, 3], 'y': [3, 4, 5]},
        ]

        frames = AnimationEngine.create_frames(states, frame_duration=500)

        assert len(frames) == 3
        assert all(isinstance(frame, go.Frame) for frame in frames)

    def test_add_play_button(self, sample_figure):
        """Test adding play button to figure."""
        sample_figure.frames = [
            go.Frame(data=[go.Scatter(x=[1, 2], y=[1, 2])]),
            go.Frame(data=[go.Scatter(x=[1, 2], y=[2, 3])]),
        ]

        result = AnimationEngine.add_play_button(sample_figure)

        assert 'updatemenus' in result.layout
        assert len(result.layout.updatemenus) > 0

    def test_add_slider(self, sample_figure):
        """Test adding slider to animated figure."""
        sample_figure.frames = [
            go.Frame(data=[go.Scatter(x=[1, 2], y=[1, 2])], name='frame1'),
            go.Frame(data=[go.Scatter(x=[1, 2], y=[2, 3])], name='frame2'),
        ]

        result = AnimationEngine.add_slider(
            sample_figure,
            frame_names=['2023', '2024']
        )

        assert 'sliders' in result.layout
        assert len(result.layout.sliders) > 0


# ==================== Helper Function Tests ====================

class TestTransitionHelpers:
    """Tests for transition helper functions."""

    @pytest.fixture
    def sample_figure(self):
        """Create sample figure."""
        return go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3])])

    def test_apply_transition_string(self, sample_figure):
        """Test applying transition with string type."""
        result = apply_transition(sample_figure, 'elastic', duration=800)

        assert result.layout.transition is not None

    def test_apply_transition_enum(self, sample_figure):
        """Test applying transition with enum type."""
        result = apply_transition(sample_figure, TransitionType.BOUNCE, duration=1000)

        assert result.layout.transition is not None

    def test_create_transition(self):
        """Test creating transition config."""
        config = create_transition('elastic', duration=800, easing='ease-out-bounce')

        assert config.transition_type == TransitionType.ELASTIC
        assert config.duration == 800
        assert config.easing == 'ease-out-bounce'


# ==================== GestureHandler Tests ====================

class TestGestureHandler:
    """Tests for GestureHandler."""

    def test_initialization(self):
        """Test gesture handler initialization."""
        handler = GestureHandler()

        assert handler.config is not None
        assert isinstance(handler.config, GestureConfig)

    def test_register_tap_handler(self):
        """Test registering tap handler."""
        handler = GestureHandler()
        callback_triggered = {'triggered': False}

        def tap_callback(event):
            callback_triggered['triggered'] = True

        handler.on_tap(tap_callback)

        # Trigger event
        event = TouchEvent(GestureType.TAP, x=100, y=200, timestamp=0.0)
        handler.handle_event(event)

        assert callback_triggered['triggered'] is True

    def test_register_pinch_handler(self):
        """Test registering pinch handler."""
        handler = GestureHandler()
        pinch_scale = {'scale': None}

        def pinch_callback(event):
            pinch_scale['scale'] = event.scale

        handler.on_pinch(pinch_callback)

        # Trigger pinch event
        event = TouchEvent(GestureType.PINCH, x=100, y=200, timestamp=0.0, scale=1.5)
        handler.handle_event(event)

        assert pinch_scale['scale'] == 1.5

    def test_register_swipe_handler(self):
        """Test registering swipe handler."""
        handler = GestureHandler()
        swipe_delta = {'delta_x': None}

        def swipe_callback(event):
            swipe_delta['delta_x'] = event.delta_x

        handler.on_swipe(swipe_callback)

        # Trigger swipe event
        event = TouchEvent(GestureType.SWIPE, x=100, y=200, timestamp=0.0, delta_x=50)
        handler.handle_event(event)

        assert swipe_delta['delta_x'] == 50

    def test_method_chaining(self):
        """Test method chaining for handlers."""
        handler = GestureHandler()

        result = (handler
                  .on_tap(lambda e: None)
                  .on_pinch(lambda e: None)
                  .on_swipe(lambda e: None))

        assert result is handler


class TestGestureConfig:
    """Tests for GestureConfig."""

    def test_default_config(self):
        """Test default gesture configuration."""
        config = GestureConfig()

        assert GestureType.TAP in config.enabled_gestures
        assert GestureType.PINCH in config.enabled_gestures

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = GestureConfig(
            tap_threshold=100,
            swipe_threshold=30
        )

        assert config.tap_threshold == 100
        assert config.swipe_threshold == 30


# ==================== Gesture Helper Tests ====================

class TestGestureHelpers:
    """Tests for gesture helper functions."""

    @pytest.fixture
    def sample_figure(self):
        """Create sample figure."""
        return go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3])])

    def test_setup_gestures(self, sample_figure):
        """Test setting up gestures on figure."""
        result = setup_gestures(sample_figure, [GestureType.TAP, GestureType.PINCH])

        assert result is not None
        assert hasattr(result, '_gesture_config')

    def test_enable_mobile_gestures(self, sample_figure):
        """Test enabling mobile gestures."""
        result = enable_mobile_gestures(
            sample_figure,
            pinch_to_zoom=True,
            tap_to_select=True
        )

        assert result is not None

    def test_optimize_for_touch(self, sample_figure):
        """Test optimizing figure for touch."""
        # Add a trace with markers
        sample_figure.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode='markers'))

        result = optimize_for_touch(sample_figure, min_touch_target=48)

        assert result is not None
        # Marker size should be increased
        if hasattr(result.data[1].marker, 'size'):
            assert result.data[1].marker.size >= 11  # 48px / 4 ≈ 12px


# ==================== TouchEvent Tests ====================

class TestTouchEvent:
    """Tests for TouchEvent."""

    def test_initialization(self):
        """Test touch event initialization."""
        event = TouchEvent(
            gesture_type=GestureType.TAP,
            x=100.0,
            y=200.0,
            timestamp=12345.0
        )

        assert event.gesture_type == GestureType.TAP
        assert event.x == 100.0
        assert event.y == 200.0
        assert event.timestamp == 12345.0

    def test_with_deltas(self):
        """Test touch event with movement deltas."""
        event = TouchEvent(
            gesture_type=GestureType.SWIPE,
            x=100.0,
            y=200.0,
            timestamp=0.0,
            delta_x=50.0,
            delta_y=-30.0
        )

        assert event.delta_x == 50.0
        assert event.delta_y == -30.0

    def test_with_scale(self):
        """Test touch event with pinch scale."""
        event = TouchEvent(
            gesture_type=GestureType.PINCH,
            x=100.0,
            y=200.0,
            timestamp=0.0,
            scale=1.5
        )

        assert event.scale == 1.5


# ==================== Integration Tests ====================

class TestAnimationsIntegration:
    """Integration tests for animations module."""

    @pytest.fixture
    def sample_figure(self):
        """Create sample figure."""
        return go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3])])

    def test_full_animation_workflow(self, sample_figure):
        """Test complete animation workflow."""
        # Create frames
        states = [
            {'x': [1, 2, 3], 'y': [i, i+1, i+2]}
            for i in range(1, 6)
        ]
        frames = AnimationEngine.create_frames(states)

        # Add frames to figure
        sample_figure.frames = frames

        # Add controls
        sample_figure = AnimationEngine.add_play_button(sample_figure)
        sample_figure = AnimationEngine.add_slider(sample_figure)

        # Apply transition
        config = TransitionConfig(duration=500, easing='ease-in-out')
        result = AnimationEngine.apply_transition(sample_figure, config)

        assert result is not None
        assert len(result.frames) == 5
        assert 'updatemenus' in result.layout
        assert 'sliders' in result.layout

    def test_mobile_optimization_workflow(self, sample_figure):
        """Test mobile optimization workflow."""
        # Enable gestures
        sample_figure = enable_mobile_gestures(sample_figure)

        # Optimize for touch
        sample_figure = optimize_for_touch(sample_figure)

        # Apply smooth transitions
        sample_figure = apply_transition(sample_figure, 'smooth', duration=300)

        assert sample_figure is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
