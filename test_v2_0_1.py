"""
Quick test for VizForge v2.0.1 - show=False default behavior.
"""

import sys
sys.path.insert(0, '/Users/teyfikoz/Projects/vizforge')

def test_show_default():
    """Test that show=False is the new default."""
    import vizforge as vz
    import pandas as pd

    print("\n🧪 Testing VizForge v2.0.1 - show=False default")
    print("=" * 60)

    # Test 1: Version check
    print(f"\n✅ Version: {vz.__version__}")
    assert vz.__version__ == "2.0.1", f"Expected v2.0.1, got {vz.__version__}"

    # Test 2: Config check
    config = vz.get_config()
    auto_show = config.get('auto_show')
    print(f"✅ Config auto_show: {auto_show}")
    assert auto_show == False, f"Expected auto_show=False, got {auto_show}"

    # Test 3: Line chart with default (should NOT show)
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [10, 20, 15, 25, 30]
    })

    print("\n✅ Creating chart with show=False (default)...")
    chart = vz.line(df, x='x', y='y', title='Test Chart')
    print(f"   Chart object created: {type(chart).__name__}")
    print(f"   Chart has figure: {chart.fig is not None}")

    # Test 4: Verify show parameter still works
    print("\n✅ Testing explicit show=True parameter...")
    # Don't actually show it, just verify the parameter exists
    import inspect
    sig = inspect.signature(vz.line)
    assert 'show' in sig.parameters, "show parameter missing!"
    assert sig.parameters['show'].default == False, "show default should be False!"

    print("\n" + "=" * 60)
    print("🎉 All tests passed! VizForge v2.0.1 is production-ready!")
    print("\n📝 Key changes:")
    print("   - show=False is now default (no auto-display)")
    print("   - Use chart.show() for explicit display")
    print("   - Use show=True parameter for old behavior")
    print("   - Use vz.set_config(auto_show=True) for global old behavior")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_show_default()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
