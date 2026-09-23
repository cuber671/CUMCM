"""py-feat 0.6.1 ↔ scipy 1.15 / numpy 2.x 兼容垫片。

用法（任何 import feat 之前）：
    import compat_feat          # noqa: F401  （注册缺失的旧符号）
    import feat

已知缺口（实测 2026-09-23）：
- nltools.analysis: `from scipy.stats import binom_test`（scipy ≥1.12 已移除）
  → 以 binomtest 重建同签名包装，返回 p 值。
- numpy 2.x 移除 np.ComplexWarning（移至 np.exceptions）→ 原位别名。
"""
import numpy as np
import scipy.stats

if not hasattr(scipy.stats, "binom_test"):
    from scipy.stats import binomtest

    def _binom_test_compat(x, n=None, p=0.5, alternative="two-sided"):
        return binomtest(x, n=n, p=p, alternative=alternative).pvalue

    scipy.stats.binom_test = _binom_test_compat

if not hasattr(np, "ComplexWarning"):
    np.ComplexWarning = np.exceptions.ComplexWarning
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
if not hasattr(np, "mat"):
    np.mat = np.asmatrix

import scipy.integrate

if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

# torchvision 0.26 移除 read_video：本项目仅用 detect_image，视频渲染走
# src/p1/media 的 ffmpeg 通道；此处提供显式降级桩保持 feat 可导入。
import torchvision.io as _tio

if not hasattr(_tio, "read_video"):
    def _read_video_compat(*args, **kwargs):
        raise NotImplementedError(
            "torchvision 0.26 已移除 read_video；本项目视频处理统一走 src/p1/media（ffmpeg/pts 通道），"
            "人脸检测用 Detector.detect_image 逐帧调用。"
        )
    _tio.read_video = _read_video_compat
