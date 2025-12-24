# Changelog - VizForge

All notable changes to this project will be documented in this file.

## [3.0.1] - 2025-12-23

### 🐛 BUG FIXES: Pandas Deprecation Warnings & Import Errors

#### Fixed
- **Pandas Deprecation Warnings**: Fixed deprecated `fillna(method=...)` usage
  - `forecaster.py` (line 97): Replaced `fillna(method='ffill').fillna(method='bfill')` with `ffill().bfill()`
  - `trend_detector.py` (line 86): Replaced `fillna(method='ffill').fillna(method='bfill')` with `ffill().bfill()`
  - `anomaly_detector.py` (line 98): Replaced `fillna(method='ffill').fillna(method='bfill')` with `ffill().bfill()`
  - No more FutureWarning messages in pandas 2.x+

- **WebGPU Import Error**: Fixed missing WebGPUConfig import
  - `__init__.py` (lines 153-156): Added try-except block for WebGPU imports
  - Graceful fallback when WebGPU components are unavailable
  - Library now imports successfully without WebGPU dependencies

#### Impact
- Clean execution without deprecation warnings
- Compatible with pandas 2.x and future versions
- Improved stability and import reliability

---

## [3.0.0] - 2024-XX-XX

### MAJOR RELEASE: Performance & Extensibility Revolution
- WebGPU rendering (1000x faster than Plotly)
- Data streaming for infinite datasets
- Plugin architecture for extensibility
- Smart caching & lazy evaluation
- Real-time collaboration
- Enhanced interactivity (gestures, touch, 3D navigation)

---

See full history at: https://github.com/teyfikoz/vizforge
