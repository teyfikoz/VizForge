"""
Video Export Demo

Demonstrates VizForge's powerful video export capabilities:
- MP4 export (H.264 encoding)
- WebM export (VP9 encoding)
- GIF export (optimized)
- Custom animations and transitions
- Progress tracking
"""

import vizforge as vz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_data():
    """Create animated sample data."""
    # Generate time series data
    dates = pd.date_range('2024-01-01', periods=12, freq='M')

    # Create multiple data frames for animation
    data_frames = []
    for month in range(1, 13):
        df = pd.DataFrame({
            'Month': dates[:month],
            'Sales': np.cumsum(np.random.randint(1000, 5000, month)),
            'Profit': np.cumsum(np.random.randint(500, 2000, month)),
            'Region': ['North', 'South', 'East', 'West'][:month % 4 + 1] * (month // 4 + 1)
        })
        data_frames.append(df[:month])

    return data_frames


def demo_mp4_export():
    """Demo 1: MP4 Video Export"""
    print("=" * 60)
    print("DEMO 1: MP4 Video Export (H.264)")
    print("=" * 60)

    # Create sample data
    data_frames = create_sample_data()

    # Create initial chart
    chart = vz.line(
        data_frames[0],
        x='Month',
        y='Sales',
        title='Monthly Sales Growth - Animated',
        theme='modern'
    )

    print("\n📹 Exporting to MP4...")
    print("   Format: H.264")
    print("   Resolution: 1920x1080")
    print("   FPS: 30")
    print("   Quality: High")

    # Progress callback
    def progress(pct):
        print(f"   Progress: {int(pct * 100)}%", end='\r')

    # Export as MP4
    from vizforge.video_export import export_video

    output_path = export_video(
        chart,
        'sales_growth.mp4',
        data_frames=data_frames,
        format='mp4',
        fps=30,
        width=1920,
        height=1080,
        quality='high',
        progress_callback=progress
    )

    print(f"\n✅ MP4 exported successfully: {output_path}")
    print(f"   Duration: ~{len(data_frames) / 30:.1f} seconds")


def demo_webm_export():
    """Demo 2: WebM Video Export"""
    print("\n" + "=" * 60)
    print("DEMO 2: WebM Video Export (VP9)")
    print("=" * 60)

    # Create bar chart data
    df = pd.DataFrame({
        'Category': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
        'Q1': [45000, 52000, 38000, 61000, 47000],
        'Q2': [48000, 55000, 42000, 63000, 51000],
        'Q3': [52000, 58000, 45000, 67000, 54000],
        'Q4': [55000, 61000, 48000, 70000, 58000],
    })

    # Create data frames for each quarter
    data_frames = []
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        df_quarter = pd.DataFrame({
            'Category': df['Category'],
            'Sales': df[quarter]
        })
        data_frames.append(df_quarter)

    # Create chart
    chart = vz.bar(
        data_frames[0],
        x='Category',
        y='Sales',
        title='Quarterly Product Sales',
        theme='professional'
    )

    print("\n📹 Exporting to WebM...")
    print("   Format: VP9")
    print("   Resolution: 1280x720")
    print("   FPS: 24")
    print("   Quality: Medium")

    from vizforge.video_export import export_video

    output_path = export_video(
        chart,
        'quarterly_sales.webm',
        data_frames=data_frames,
        format='webm',
        fps=24,
        width=1280,
        height=720,
        quality='medium'
    )

    print(f"✅ WebM exported successfully: {output_path}")


def demo_gif_export():
    """Demo 3: Animated GIF Export"""
    print("\n" + "=" * 60)
    print("DEMO 3: Animated GIF Export")
    print("=" * 60)

    # Create pie chart data showing market share evolution
    data_frames = []
    companies = ['Company A', 'Company B', 'Company C', 'Company D']

    for year in range(2020, 2025):
        df = pd.DataFrame({
            'Company': companies,
            'Market_Share': np.random.dirichlet(np.ones(4)) * 100
        })
        data_frames.append(df)

    # Create chart
    chart = vz.pie(
        data_frames[0],
        values='Market_Share',
        names='Company',
        title='Market Share Evolution (2020-2024)',
        theme='colorful'
    )

    print("\n📹 Exporting to GIF...")
    print("   Format: Optimized GIF")
    print("   Resolution: 800x600")
    print("   FPS: 2 (slower animation)")
    print("   Loop: Infinite")

    from vizforge.video_export import export_video

    output_path = export_video(
        chart,
        'market_share.gif',
        data_frames=data_frames,
        format='gif',
        fps=2,
        width=800,
        height=600,
        quality='high'
    )

    print(f"✅ GIF exported successfully: {output_path}")


def demo_advanced_animation():
    """Demo 4: Advanced Animation with Custom Transitions"""
    print("\n" + "=" * 60)
    print("DEMO 4: Advanced Animation Engine")
    print("=" * 60)

    from vizforge.video_export import (
        VideoExporter,
        VideoConfig,
        VideoFormat,
        AnimationEngine,
        AnimationType,
        FrameGenerator
    )

    # Create scatter plot data
    np.random.seed(42)
    n_points = 50

    data_frames = []
    for t in range(10):
        angle = t * np.pi / 5
        x = np.random.randn(n_points) + np.cos(angle) * 3
        y = np.random.randn(n_points) + np.sin(angle) * 3
        df = pd.DataFrame({
            'X': x,
            'Y': y,
            'Size': np.random.randint(10, 100, n_points)
        })
        data_frames.append(df)

    # Create chart
    chart = vz.scatter(
        data_frames[0],
        x='X',
        y='Y',
        size='Size',
        title='Animated Scatter Plot - Elastic Transition',
        theme='dark'
    )

    print("\n🎬 Creating video with custom animations...")
    print("   Animation: Elastic (spring-like)")
    print("   Transition: Smooth")

    # Configure video with custom settings
    config = VideoConfig(
        format=VideoFormat.MP4,
        fps=30,
        duration=5.0,
        width=1280,
        height=720,
        quality='high',
        loop=True,
        optimize=True
    )

    exporter = VideoExporter(chart, config)
    output_path = exporter.export(
        'scatter_elastic.mp4',
        data_frames=data_frames,
        transition='smooth'
    )

    print(f"✅ Advanced animation exported: {output_path}")


def demo_frame_generator():
    """Demo 5: Frame Generator and Custom Processing"""
    print("\n" + "=" * 60)
    print("DEMO 5: Frame Generator - Custom Processing")
    print("=" * 60)

    from vizforge.video_export import FrameGenerator

    # Create simple data
    df1 = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D'],
        'Value': [10, 20, 15, 25]
    })

    df2 = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D'],
        'Value': [25, 15, 30, 20]
    })

    # Create chart
    chart = vz.bar(df1, x='Category', y='Value', title='Frame Generator Demo')

    print("\n🎞️ Generating individual frames...")

    # Create frame generator
    generator = FrameGenerator(chart, width=800, height=600)

    # Generate interpolated frames
    frames = generator.generate_interpolated_frames(
        df1, df2,
        n_frames=30,
        easing_function=lambda t: t * t  # Ease-in
    )

    print(f"   Generated {len(frames)} frames")

    # Save frames
    frame_paths = generator.save_frames(
        'output_frames',
        prefix='demo_frame',
        format='png'
    )

    print(f"   Saved {len(frame_paths)} PNG files")
    print(f"   Memory usage: ~{generator.estimate_memory_usage() / 1024 / 1024:.1f} MB")

    # Add watermark
    print("\n   Adding watermark...")
    generator.add_watermark("VizForge Demo", opacity=180)

    # Optimize
    print("   Optimizing frames...")
    generator.optimize_frames(quality=85, resize_factor=0.8)

    print("✅ Frame generation complete!")


def demo_animation_types():
    """Demo 6: Different Animation Types Comparison"""
    print("\n" + "=" * 60)
    print("DEMO 6: Animation Types Comparison")
    print("=" * 60)

    from vizforge.video_export import AnimationType

    animation_types = [
        ('smooth', 'Smooth (Ease In-Out Cubic)'),
        ('elastic', 'Elastic (Spring-like)'),
        ('bounce', 'Bounce'),
        ('ease_in', 'Ease In'),
        ('ease_out', 'Ease Out')
    ]

    # Create sample data
    df_start = pd.DataFrame({
        'X': [1, 2, 3, 4, 5],
        'Y': [2, 4, 3, 5, 4]
    })

    df_end = pd.DataFrame({
        'X': [1, 2, 3, 4, 5],
        'Y': [5, 3, 6, 2, 7]
    })

    print("\n🎨 Creating videos with different animation types...")

    from vizforge.video_export import export_video

    for anim_type, description in animation_types:
        print(f"\n   • {description}")

        chart = vz.line(
            df_start,
            x='X',
            y='Y',
            title=f'Animation: {description}'
        )

        output_path = export_video(
            chart,
            f'animation_{anim_type}.mp4',
            data_frames=[df_start, df_end],
            format='mp4',
            fps=30,
            width=800,
            height=600,
            quality='medium'
        )

        print(f"     ✓ Saved: {output_path}")

    print("\n✅ All animation types exported!")


def demo_performance_benchmark():
    """Demo 7: Performance Benchmarking"""
    print("\n" + "=" * 60)
    print("DEMO 7: Performance Benchmark")
    print("=" * 60)

    import time
    from vizforge.video_export import VideoExporter, VideoConfig, VideoFormat

    # Create large dataset
    print("\n⚡ Testing with large dataset...")
    data_frames = []
    for i in range(20):
        df = pd.DataFrame({
            'X': np.random.randn(1000),
            'Y': np.random.randn(1000),
            'Category': np.random.choice(['A', 'B', 'C'], 1000)
        })
        data_frames.append(df)

    chart = vz.scatter(
        data_frames[0],
        x='X',
        y='Y',
        color='Category',
        title='Performance Test - 1000 Points per Frame'
    )

    # Benchmark
    start_time = time.time()

    config = VideoConfig(
        format=VideoFormat.MP4,
        fps=30,
        width=1920,
        height=1080,
        quality='medium'
    )

    exporter = VideoExporter(chart, config)
    output_path = exporter.export('benchmark.mp4', data_frames=data_frames)

    elapsed = time.time() - start_time

    print(f"\n📊 Performance Results:")
    print(f"   Frames: {len(data_frames)}")
    print(f"   Points per frame: 1000")
    print(f"   Total time: {elapsed:.2f} seconds")
    print(f"   FPS (generation): {len(data_frames) / elapsed:.1f} frames/sec")
    print(f"   Output: {output_path}")

    print("\n✅ Benchmark complete!")


def main():
    """Run all video export demos."""
    print("\n" + "=" * 70)
    print("🎬 VizForge Video Export Engine - Comprehensive Demo")
    print("=" * 70)

    print("\nSupported Formats:")
    print("  • MP4 (H.264) - Best for presentations & social media")
    print("  • WebM (VP9) - Best for web embedding")
    print("  • GIF - Best for emails & quick sharing")

    print("\nFeatures:")
    print("  • Smooth animations with easing functions")
    print("  • Progress tracking")
    print("  • Customizable FPS, resolution, quality")
    print("  • Frame interpolation")
    print("  • Watermark support")
    print("  • Memory optimization")

    try:
        # Run demos
        demo_mp4_export()
        demo_webm_export()
        demo_gif_export()
        demo_advanced_animation()
        demo_frame_generator()
        demo_animation_types()
        demo_performance_benchmark()

        print("\n" + "=" * 70)
        print("✅ ALL VIDEO EXPORT DEMOS COMPLETE!")
        print("=" * 70)

        print("\n🚀 Quick Start:")
        print("""
import vizforge as vz
from vizforge.video_export import export_video

# Create chart
chart = vz.line(df, x='date', y='sales')

# Export as video
export_video(
    chart,
    'output.mp4',
    data_frames=[df1, df2, df3],  # Animation frames
    fps=30,
    quality='high'
)
        """)

        print("\n📚 Documentation:")
        print("   • MP4: Requires ffmpeg (install from ffmpeg.org)")
        print("   • WebM: Requires ffmpeg (VP9 codec)")
        print("   • GIF: Requires Pillow (pip install Pillow)")

    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("\n💡 Note: Video export requires additional dependencies:")
        print("   • ffmpeg (for MP4/WebM): https://ffmpeg.org/download.html")
        print("   • Pillow (for GIF): pip install Pillow")
        print("\nThese demos show the API even if dependencies are missing.")


if __name__ == '__main__':
    main()
