#!/usr/bin/env python3
"""
Web版相机显示应用
使用Flask在浏览器中显示机器人相机图像
"""

import time
import threading
import base64
import io
import re
import struct
import numpy as np
import agibot_gdk
from camera_map import gdk_camera_entries, GDK_TYPE_BY_CONF_NAME, hand_rgbd_mode
from typing import Optional, Dict, Any, List
import os
import json
import signal
import sys
import subprocess
import importlib.util
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 头部双目内外参：使用工程内 sg_standalone.py（G2 接线）
SG_STANDALONE_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sg_standalone.py"),
    os.path.expanduser("~/sg_standalone.py"),
    "/home/agi/sg_standalone.py",
]
# G2 头部双目 I2C（与实测可用配置一致）
SG_I2C_BUS = 9
SG_LEFT_ADDR = 0x3C
SG_RIGHT_ADDR = 0x3B
SG_DES_ADDR = 0x29
I2C_DEV_PATH = f"/dev/i2c-{SG_I2C_BUS}"

# 尝试导入Flask
try:
    from flask import Flask, render_template_string, jsonify, Response, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("⚠️ Flask未安装，将使用简化模式")

# 尝试导入OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("⚠️ OpenCV未安装，将使用原始数据模式")

# D2C PnP / 深度平面指标（同目录模块）
try:
    from d2c_plane_metrics import (
        evaluate_d2c_plane,
        find_intrinsic_file,
        load_intrinsic_json,
        metrics_for_json,
        parse_intrinsic,
    )
    HAS_D2C_PLANE = True
except ImportError:
    HAS_D2C_PLANE = False
    print("⚠️ d2c_plane_metrics 未加载，平面精度一键测不可用")

class WebCameraViewer:
    """Web版相机查看器"""
    
    def __init__(self, host='0.0.0.0', port=5000):
        """初始化Web相机查看器"""
        self.camera = None
        self.host = host
        self.port = port
        self.current_image = None
        self.current_camera_index = 0
        self.is_running = False
        self.app = None
        self._cleanup_lock = threading.Lock()
        
        # 图像缓存 - 使用更智能的缓存策略
        self.image_cache = {}  # {camera_type: (image_b64, timestamp, image_timestamp)}
        self.cache_timeout = 0.1  # 100ms缓存超时，减少重复处理
        self.last_image_timestamps = {}  # 记录每个相机的最后图像时间戳
        # 远程预览：深度伪彩较重，控制预览宽度避免卡死
        self.preview_max_width = 640
        self.preview_jpeg_quality = 70
        self.preview_fetch_timeout_ms = 40.0
        # /frame 短缓存，挡住前端连发重复请求；失败也短缓存，避免无帧时 1s 超时连打卡死
        self._preview_jpeg_cache: Dict[int, tuple] = {}
        self._preview_jpeg_cache_ttl = 0.12
        self._preview_miss_cache_ttl = 0.35
        self._preview_jpeg_lock = threading.Lock()
        # 手部深度预览：原生未对齐 1280x960；对齐画布可能是 1280x1056，规整到 out_size
        self.hand_depth_native_size = (1280, 960)
        self.hand_depth_out_size = (1280, 960)  # (w, h) 预览用
        # 点云模式：
        #   d2c = 深度已对齐到 RGB（1280x1056），必须用 Intr_RGB 反投影
        #   tof = 原生 TOF（640x480），用 Intr_TOF
        # 若用 TOF 内参去解 D2C 深度，会出现典型「锥形」畸变
        self.hand_pc_mode = 'd2c'
        self.hand_pc_depth_size = (640, 480)  # 仅 tof 模式使用
        # 点云绕光轴(Z)旋转 180°：x,y 取反，与手部安装朝向对齐（预览/落盘一致）
        self.hand_pc_rotate_z_deg = 180
        # 深度预览：与 dds_camera_view 对齐（固定 200–3000mm + JET）
        self.depth_vis_z_min_mm = 200.0
        self.depth_vis_z_max_mm = 3000.0
        # 0 / 65535 为无效；其余按显示量程裁剪上色
        self.depth_vis_valid_min_mm = 1.0
        self.depth_vis_valid_max_mm = 65534.0
        # None=固定 z_min/z_max（与 DDS 预览一致）；可设 (5,95) 做百分位自适应
        self.depth_vis_auto_percentile = None
        self.depth_vis_auto_min_span_mm = 50.0
        # 深度伪彩：与 dds_camera_view 一致用 jet
        self.depth_vis_colormap = 'jet'
        self._info_cache = {}  # camera_type -> (text, ts)
        self._info_cache_ttl = 1.0
        self.save_dir = os.path.expanduser("~/camera_captures")
        os.makedirs(self.save_dir, exist_ok=True)
        # 采集工程：按时长开启，时段内多次采集共用同一工程目录
        self._project_lock = threading.Lock()
        self.active_project = None  # dict: name/dir/start/end/duration/count/...
        self.default_capture_duration_sec = 300  # 默认采集时长 5 分钟
        # 手部 RGBD 点云连续录制状态
        self._pc_record_lock = threading.Lock()
        self._pc_record = {
            'active': False,
            'thread': None,
            'side': None,       # left / right
            'index': None,
            'dir': None,
            'count': 0,
            'interval_s': 0.1,
            'pixel_stride': 2,  # 连续录制降采样，减小体积
            'stop_event': None,
        }
        # 无工厂内参时的水平视场角回退（度）
        self.hand_depth_hfov_deg = 75.0
        self.hand_intrinsic_search_dirs = [
            os.path.expanduser('~/camera_intrinsics'),
            os.path.expanduser('~/calibration'),
            '/data/calibration',
            '/home/agi/app/conf/camera_params',
            os.path.join(self.save_dir, 'intrinsics'),
            '/home/agi/camera_intrinsics',
        ]
        self._hand_intr_cache = {}  # (side, width, height) -> intrinsic dict
        self._hand_intr_gdk_tried = set()  # sides already attempted via GDK
        
        # 设置信号处理 
        self.setup_signal_handlers()

        self.camera_product = os.environ.get("CAMERA_PRODUCT")
        self.hand_rgbd = hand_rgbd_mode(self.camera_product)
        self.camera_types = self._build_camera_types()
        print(f"机型 CAMERA_PRODUCT={self.camera_product or '(默认)'} "
              f"hand_rgbd={self.hand_rgbd} 相机数={len(self.camera_types)}")
        for i, (_, name) in enumerate(self.camera_types):
            print(f"  [{i}] {name}")
        
        # 初始化相机
        self.init_camera()
        
        # 初始化Flask应用
        if HAS_FLASK:
            self.app = Flask(__name__)
            # 配置Flask以安全退出
            self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
            self.app.config['TEMPLATES_AUTO_RELOAD'] = False
            self.app.config['EXPLAIN_TEMPLATE_LOADING'] = False
            self.setup_routes()
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        self._cleanup_done = False
        
        def signal_handler(signum, frame):
            if self._cleanup_done:
                print(f"\n⚠️ 强制退出...")
                sys.exit(1)
            
            print(f"\n⚠️ 收到信号 {signum}，正在安全退出...")
            self._cleanup_done = True
            self.cleanup()
            sys.exit(0)
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
    
    def cleanup(self):
        """清理资源"""
        with self._cleanup_lock:
            if hasattr(self, '_cleanup_done') and self._cleanup_done:
                return
                
            print("🧹 正在清理资源...")
            self.is_running = False
            
            # 关闭相机
            try:
                self.stop_hand_pointcloud_recording()
            except Exception:
                pass
            if self.camera:
                try:
                    self.camera.close_camera()
                    print("✅ 相机已关闭")
                except Exception as e:
                    print(f"⚠️ 关闭相机时出错: {e}")
                finally:
                    self.camera = None

            if hasattr(agibot_gdk, "gdk_release"):
                try:
                    agibot_gdk.gdk_release()
                except Exception as e:
                    print(f"⚠️ gdk_release 出错: {e}")
            
            # 清理图像数据
            self.current_image = None
            
            # 等待一小段时间确保资源释放
            time.sleep(0.1)
            
            print("✅ 资源清理完成")
    
    def init_camera(self):
        """初始化相机"""
        try:
            print("正在初始化相机...")
            if hasattr(agibot_gdk, "gdk_init"):
                res = agibot_gdk.gdk_init()
                if hasattr(agibot_gdk, "GDKRes") and res != agibot_gdk.GDKRes.kSuccess:
                    print(f"⚠️ GDK 初始化返回: {res}")
                else:
                    print(f"GDK 初始化: {res}")
            self.camera = agibot_gdk.Camera()
            time.sleep(3)  # 等待相机初始化
            
            print("相机初始化完成！")
        except Exception as e:
            print(f"相机初始化失败: {e}")
            self.camera = None

    def _build_camera_types(self) -> List[tuple]:
        """按 camera_device_map.json + CAMERA_PRODUCT 构建 GDK 相机列表。"""
        entries = gdk_camera_entries(self.camera_product)
        types: List[tuple] = []
        for conf_name, label in entries:
            enum_name = GDK_TYPE_BY_CONF_NAME.get(conf_name)
            if not enum_name:
                continue
            ct = getattr(agibot_gdk.CameraType, enum_name, None)
            if ct is None:
                print(f"⚠️ GDK 无 CameraType.{enum_name}，跳过 {conf_name}")
                continue
            types.append((ct, label))
        if types:
            return types
        # 回退：与旧版硬编码一致
        print("⚠️ gdk_camera_entries 为空，使用硬编码相机列表")
        return [
            (agibot_gdk.CameraType.kHeadStereoLeft, "头部双目左相机"),
            (agibot_gdk.CameraType.kHeadStereoRight, "头部双目右相机"),
            (agibot_gdk.CameraType.kHandLeftColor, "左手彩色相机"),
            (agibot_gdk.CameraType.kHandRightColor, "右手彩色相机"),
            (agibot_gdk.CameraType.kHeadDepth, "头部深度相机"),
            (agibot_gdk.CameraType.kHeadColor, "头部彩色相机"),
        ]


    def decode_image_data(self, image, for_preview: bool = False) -> Optional[np.ndarray]:
        """解码图像数据。for_preview=True 时用更轻量的深度可视化。"""
        if not hasattr(image, 'data') or image.data is None or len(image.data) == 0:
            return None
        #print(f"解码图像数据: {image.encoding} {image.color_format} 尺寸: {image.width}x{image.height}")
        try:
            # 根据编码格式解码数据
            if image.encoding == agibot_gdk.Encoding.JPEG:
                if HAS_OPENCV:
                    nparr = np.frombuffer(image.data, np.uint8)
                    decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return decoded_image
                else:
                    # 直接返回原始JPEG数据
                    return image.data
                    
            elif image.encoding == agibot_gdk.Encoding.PNG:
                if HAS_OPENCV:
                    nparr = np.frombuffer(image.data, np.uint8)
                    decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return decoded_image
                else:
                    # 直接返回原始PNG数据
                    return image.data
                    
            elif image.encoding == agibot_gdk.Encoding.UNCOMPRESSED:
                # 未压缩数据，直接转换
                if image.color_format == agibot_gdk.ColorFormat.RGB:
                    decoded_image = np.frombuffer(image.data, dtype=np.uint8)
                    decoded_image = decoded_image.reshape((image.height, image.width, 3))
                    if HAS_OPENCV:
                        # RGB转BGR
                        decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGR)
                    return decoded_image
                elif image.color_format == agibot_gdk.ColorFormat.BGR:
                    decoded_image = np.frombuffer(image.data, dtype=np.uint8)
                    decoded_image = decoded_image.reshape((image.height, image.width, 3))
                    return decoded_image
                elif image.color_format == agibot_gdk.ColorFormat.GRAY8:
                    decoded_image = np.frombuffer(image.data, dtype=np.uint8)
                    decoded_image = decoded_image.reshape((image.height, image.width))
                    if HAS_OPENCV:
                        # 灰度转BGR
                        decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_GRAY2BGR)
                    return decoded_image
                elif image.color_format == agibot_gdk.ColorFormat.GRAY16:
                    decoded_image = np.frombuffer(image.data, dtype=np.uint16)
                    decoded_image = decoded_image.reshape((image.height, image.width))
                    # 转换为8位
                    decoded_image = (decoded_image / 256).astype(np.uint8)
                    if HAS_OPENCV:
                        decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_GRAY2BGR)
                    return decoded_image
                elif image.color_format == agibot_gdk.ColorFormat.RS2_FORMAT_Z16:
                    # Intel RealSense 16位深度；metadata 宽高可能与 buffer 不一致
                    raw = np.frombuffer(image.data, dtype=np.uint16)
                    depth = self._reshape_depth_u16(
                        raw, int(image.width or 0), int(image.height or 0))
                    depth = self._prepare_hand_depth_u16(depth)
                    return self._colorize_depth_u16(depth, for_preview=for_preview)
            
            return None
                
        except Exception as e:
            print(f"解码图像数据失败: {e}")
            return None

    def _reshape_depth_u16(
        self, raw: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """将 Z16 buffer 规整为 HxW；优先用实际字节推断尺寸。"""
        raw = np.asarray(raw, dtype=np.uint16).reshape(-1)
        w = int(width) if width else 0
        h = int(height) if height else 0
        need = w * h
        # meta 可能被改成目标输出尺寸（如 640x480），但 buffer 仍是对齐后的 1280x1056
        known = ((1280, 1056), (1280, 960), (640, 480), (640, 400), (2560, 960))
        if need <= 0 or raw.size != need:
            for kw, kh in known:
                if raw.size == kw * kh:
                    if (w, h) != (kw, kh):
                        print(
                            f"⚠️ Z16 尺寸修正: meta={w}x{h} -> buffer={kw}x{kh}",
                            flush=True,
                        )
                    w, h = kw, kh
                    need = w * h
                    break
        if need > 0 and raw.size == need:
            depth = raw.reshape((h, w))
        elif w > 0 and raw.size % w == 0:
            h2 = raw.size // w
            if h2 != h:
                print(f"⚠️ Z16 尺寸修正: meta={w}x{h} -> actual={w}x{h2}")
            depth = raw.reshape((h2, w))
        elif h > 0 and raw.size % h == 0:
            w2 = raw.size // h
            if w2 != w:
                print(f"⚠️ Z16 尺寸修正: meta={w}x{h} -> actual={w2}x{h}")
            depth = raw.reshape((h, w2))
        else:
            raise ValueError(
                f'Z16 buffer 与宽高不匹配: meta={w}x{h}, u16={raw.size}'
            )
        return self._deinterleave_packed_depth_u16(depth)

    def _deinterleave_packed_depth_u16(self, depth: np.ndarray) -> np.ndarray:
        """
        左手深度在部分模组/ISP 异常时会把「深度 + 辅助通道」按列交织进
        1280 宽 buffer（传感器原始 2560x960 Bayer 字节数不变），若整幅当 Z16
        显示会严重错分辨率/花屏。检测后取深度列并横向 x2 还原 FOV 宽度。
        """
        if depth is None or depth.ndim != 2:
            return depth
        h, w = depth.shape
        if w < 4 or w % 2 != 0:
            return depth

        # 相邻列若呈「大/小」高频交织，水平梯度会异常大
        d = depth.astype(np.float32)
        gx = float(np.mean(np.abs(d[:, 1:] - d[:, :-1])))
        gy = float(np.mean(np.abs(d[1:, :] - d[:-1, :]))) if h > 1 else 0.0

        even = depth[:, 0::2]
        odd = depth[:, 1::2]

        def _channel_score(ch: np.ndarray) -> float:
            valid = ch[(ch > 50) & (ch < 5000)]
            if valid.size < 500:
                return -1e9
            cf = ch.astype(np.float32)
            ch_gx = float(np.mean(np.abs(cf[:, 1:] - cf[:, :-1]))) if ch.shape[1] > 1 else 1e9
            return float(valid.size) / (ch_gx + 1.0)

        s_even, s_odd = _channel_score(even), _channel_score(odd)
        best_score = max(s_even, s_odd)
        worse_score = min(s_even, s_odd)
        # 条件：整体水平梯度很大，且某一列通道明显更像真实深度
        interleaved = (
            gx > 200.0
            and best_score > 100.0
            and (worse_score < 0 or best_score > 5.0 * max(worse_score, 1.0))
        )
        if not interleaved:
            return depth

        best = odd if s_odd >= s_even else even
        # 按列取深度后，最近邻横向 x2 还原 FOV 宽度（保持原始高度，勿拉到 1056）
        if HAS_OPENCV:
            restored = cv2.resize(best, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            restored = np.repeat(best, 2, axis=1)
        msg = (
            f"⚠️ Z16 交织深度已拆分: {w}x{h} -> {best.shape[1]}x{best.shape[0]} "
            f"-> restore {restored.shape[1]}x{restored.shape[0]} "
            f"(use={'odd' if s_odd >= s_even else 'even'} cols, "
            f"gx={gx:.0f}/{gy:.0f}, score={best_score:.1f}/{worse_score:.1f})"
        )
        print(msg, flush=True)
        return restored

    def _normalize_hand_depth_size(self, depth: np.ndarray) -> np.ndarray:
        """规整到手部真实/目标深度分辨率（默认 1280x960）。"""
        if depth is None or depth.ndim != 2 or not HAS_OPENCV:
            return depth
        tw, th = self.hand_depth_out_size
        if tw <= 0 or th <= 0:
            return depth
        h, w = depth.shape[:2]
        if w == tw and h == th:
            return depth
        # 对齐画布 1280x1056 → 原生 1280x960：优先垂直居中裁，避免拉伸变形
        if w == tw and h > th:
            y0 = (h - th) // 2
            return depth[y0:y0 + th, :].copy()
        if w == tw and h < th:
            # 不足高度则 pad（少见）
            out = np.zeros((th, tw), dtype=depth.dtype)
            y0 = (th - h) // 2
            out[y0:y0 + h, :] = depth
            return out
        return cv2.resize(depth, (tw, th), interpolation=cv2.INTER_NEAREST)

    def _prepare_hand_depth_u16(self, depth: np.ndarray) -> np.ndarray:
        """手部深度后处理：规整到 1280x960（不做有效 ROI 动态裁剪）。"""
        if depth is None:
            return depth
        nw, nh = self.hand_depth_native_size
        if depth.shape[1] == nw and depth.shape[0] == nh:
            return depth
        return self._normalize_hand_depth_size(depth)

    def _extract_depth_u16(
        self, image, normalize_hand: bool = True
    ) -> Optional[np.ndarray]:
        """从 GDK Image 提取 Z16 HxW；头部 D2C 时 normalize_hand=False 保持原生尺寸。"""
        if image is None or not hasattr(image, 'data') or image.data is None:
            return None
        if len(image.data) == 0:
            return None
        try:
            if image.encoding != agibot_gdk.Encoding.UNCOMPRESSED:
                return None
            if image.color_format != agibot_gdk.ColorFormat.RS2_FORMAT_Z16:
                return None
            raw = np.frombuffer(image.data, dtype=np.uint16)
            depth = self._reshape_depth_u16(
                raw, int(image.width or 0), int(image.height or 0))
            if normalize_hand:
                depth = self._prepare_hand_depth_u16(depth)
            return depth
        except Exception as e:
            print(f"提取深度失败: {e}")
            return None

    def _colorize_depth_u16(
        self,
        depth_u16: np.ndarray,
        for_preview: bool = False,
        z_min_mm: Optional[float] = None,
        z_max_mm: Optional[float] = None,
        draw_label: bool = True,
        return_valid: bool = False,
        auto_percentile: Optional[tuple] = None,
    ):
        """
        深度伪彩，默认与 dds_camera_view.depth_to_jpeg 一致：
        - 无效值：0 与 65535
        - 固定显示量程 200–3000 mm（可传 z_min/z_max 覆盖）
        - COLORMAP_JET
        - 标签：Z16 min-max mm (display lo-hi)
        """
        depth = np.asarray(depth_u16, dtype=np.uint16)
        if depth.ndim != 2:
            raise ValueError('depth_u16 必须是 HxW')

        if for_preview and HAS_OPENCV:
            h, w = depth.shape[:2]
            max_w = int(self.preview_max_width)
            if w > max_w:
                scale = max_w / float(w)
                nh = max(1, int(round(h * scale)))
                depth = cv2.resize(depth, (max_w, nh), interpolation=cv2.INTER_NEAREST)

        # 与 DDS 预览一致：0 / 65535 为无效
        valid = (depth > 0) & (depth < np.iinfo(np.uint16).max)
        # 可选再叠一层有效带（默认 1–65534，等价于上面）
        v_lo = float(self.depth_vis_valid_min_mm)
        v_hi = float(self.depth_vis_valid_max_mm)
        if v_lo > 1.0 or v_hi < 65534.0:
            depth_f_tmp = depth.astype(np.float32)
            valid = valid & (depth_f_tmp >= v_lo) & (depth_f_tmp <= v_hi)

        if not np.any(valid):
            empty = (
                np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
                if HAS_OPENCV else np.zeros(depth.shape, dtype=np.uint8)
            )
            if draw_label and HAS_OPENCV:
                lower = max(0, int(self.depth_vis_z_min_mm if z_min_mm is None else z_min_mm))
                upper = max(lower + 1, int(self.depth_vis_z_max_mm if z_max_mm is None else z_max_mm))
                cv2.putText(
                    empty,
                    f'Z16 no valid depth (display {lower}-{upper} mm)',
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 if for_preview else 0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            return (empty, valid) if return_valid else empty

        depth_f = depth.astype(np.float32)
        pct = (
            auto_percentile
            if auto_percentile is not None
            else self.depth_vis_auto_percentile
        )
        use_auto = z_min_mm is None and z_max_mm is None and pct is not None
        if use_auto:
            p_lo, p_hi = float(pct[0]), float(pct[1])
            if p_hi <= p_lo:
                p_lo, p_hi = 5.0, 95.0
            vals = depth_f[valid]
            z_lo = float(np.percentile(vals, p_lo))
            z_hi = float(np.percentile(vals, p_hi))
            min_span = float(self.depth_vis_auto_min_span_mm)
            if z_hi - z_lo < min_span:
                mid = 0.5 * (z_lo + z_hi)
                z_lo = mid - 0.5 * min_span
                z_hi = mid + 0.5 * min_span
            mode_tag = f'p{int(p_lo)}-{int(p_hi)}'
        else:
            z_lo = float(self.depth_vis_z_min_mm if z_min_mm is None else z_min_mm)
            z_hi = float(self.depth_vis_z_max_mm if z_max_mm is None else z_max_mm)
            mode_tag = 'fixed'

        lower = max(0, int(round(z_lo)))
        upper = max(lower + 1, int(round(z_hi)))
        clipped = np.clip(depth, lower, upper)
        normalized = ((clipped.astype(np.float32) - lower) * (255.0 / (upper - lower)))
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        normalized[~valid] = 0

        if not HAS_OPENCV:
            return (normalized, valid) if return_valid else normalized

        cmap = self._resolve_depth_colormap(self.depth_vis_colormap)
        colored = cv2.applyColorMap(normalized, cmap)
        colored[~valid] = (0, 0, 0)
        if draw_label:
            values = depth[valid]
            label = (
                f'Z16 {int(values.min())}-{int(values.max())} mm '
                f'(display {lower}-{upper})'
            )
            if mode_tag != 'fixed':
                label += f'/{mode_tag}'
            cv2.putText(
                colored,
                label,
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55 if for_preview else 0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return (colored, valid) if return_valid else colored

    def _resolve_depth_colormap(self, name: Optional[str]):
        """将名称映射到 OpenCV colormap，缺失时按 jet→turbo→inferno 回退。"""
        key = str(name or 'jet').strip().lower()
        aliases = {
            'inferno': 'COLORMAP_INFERNO',
            'magma': 'COLORMAP_MAGMA',
            'jet': 'COLORMAP_JET',
            'turbo': 'COLORMAP_TURBO',
            'viridis': 'COLORMAP_VIRIDIS',
            'plasma': 'COLORMAP_PLASMA',
        }
        order = [key] + [k for k in ('jet', 'turbo', 'inferno', 'magma') if k != key]
        for k in order:
            attr = aliases.get(k)
            if attr and hasattr(cv2, attr):
                return getattr(cv2, attr)
        return cv2.COLORMAP_JET

    def build_head_d2c_overlay(
        self,
        alpha: float = 0.45,
        z_min_mm: float = 200.0,
        z_max_mm: float = 3000.0,
        mode: str = 'overlay',
        edges: bool = False,
        timeout_ms: float = 120.0,
    ) -> Dict[str, Any]:
        """
        头部彩色 + 深度 D2C 可视化。
        GDK 无独立 align API；当前两路同为 640x400，按像素叠加验对齐。
        """
        if self.camera is None:
            return {'success': False, 'error': '相机未初始化'}
        if not HAS_OPENCV:
            return {'success': False, 'error': '需要 OpenCV'}

        alpha = float(np.clip(alpha, 0.0, 1.0))
        mode = (mode or 'overlay').strip().lower()
        if mode not in ('overlay', 'side', 'color'):
            mode = 'overlay'

        try:
            color_img = self.camera.get_latest_image(
                agibot_gdk.CameraType.kHeadColor, float(timeout_ms))
            depth_img = self.camera.get_latest_image(
                agibot_gdk.CameraType.kHeadDepth, float(timeout_ms))
        except Exception as e:
            return {'success': False, 'error': f'取帧失败: {e}'}

        if color_img is None or depth_img is None:
            return {'success': False, 'error': '头部彩色或深度为空'}

        color = self.decode_image_data(color_img, for_preview=False)
        depth_u16 = self._extract_depth_u16(depth_img, normalize_hand=False)
        if color is None or depth_u16 is None:
            return {'success': False, 'error': '头部彩色/深度解码失败'}
        if color.ndim == 2:
            color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
        elif color.shape[2] == 4:
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)

        ch, cw = color.shape[:2]
        dh, dw = depth_u16.shape[:2]
        size_matched = (dh, dw) == (ch, cw)
        if not size_matched:
            depth_u16 = cv2.resize(
                depth_u16, (cw, ch), interpolation=cv2.INTER_NEAREST)

        depth_vis, valid = self._colorize_depth_u16(
            depth_u16,
            for_preview=False,
            z_min_mm=z_min_mm,
            z_max_mm=z_max_mm,
            draw_label=False,
            return_valid=True,
        )

        if mode == 'color':
            out = color.copy()
        else:
            blend = color.copy()
            if np.any(valid):
                a = alpha
                overlay_region = (
                    color[valid].astype(np.float32) * (1.0 - a)
                    + depth_vis[valid].astype(np.float32) * a
                )
                blend[valid] = np.clip(overlay_region, 0, 255).astype(np.uint8)

            if edges and np.any(valid):
                edge_src = (valid.astype(np.uint8) * 255)
                edge = cv2.Canny(edge_src, 50, 150)
                blend[edge > 0] = (0, 255, 255)

            label = f'D2C a={alpha:.2f} z={int(z_min_mm)}-{int(z_max_mm)}mm'
            cv2.putText(
                blend, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)

            if mode == 'side':
                gap = 8
                panel_h = ch
                panel_w = cw
                canvas = np.zeros(
                    (panel_h, panel_w * 3 + gap * 2, 3), dtype=np.uint8)
                canvas[:, 0:panel_w] = color
                canvas[:, panel_w + gap:panel_w * 2 + gap] = depth_vis
                canvas[:, panel_w * 2 + gap * 2:] = blend
                for i, title in enumerate(('Color', 'Depth', 'D2C')):
                    x0 = i * (panel_w + gap) + 10
                    cv2.putText(
                        canvas, title, (x0, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
                out = canvas
            else:
                out = blend

        out = self.prepare_preview_frame(out)
        jpeg = self.encode_preview_jpeg(out)
        if not jpeg:
            return {'success': False, 'error': 'JPEG 编码失败'}

        color_ts = getattr(color_img, 'timestamp', None)
        depth_ts = getattr(depth_img, 'timestamp', None)
        try:
            dt_ms = (
                abs(float(color_ts) - float(depth_ts)) * 1000.0
                if color_ts is not None and depth_ts is not None else None
            )
        except Exception:
            dt_ms = None

        valid_ratio = float(np.count_nonzero(valid)) / float(valid.size)
        return {
            'success': True,
            'jpeg': jpeg,
            'meta': {
                'color_size': [int(cw), int(ch)],
                'depth_native_size': [int(dw), int(dh)],
                'aligned_size': [int(cw), int(ch)],
                'size_matched': bool(size_matched),
                'valid_depth_ratio': round(valid_ratio, 4),
                'alpha': alpha,
                'z_min_mm': float(z_min_mm),
                'z_max_mm': float(z_max_mm),
                'mode': mode,
                'edges': bool(edges),
                'color_timestamp': color_ts,
                'depth_timestamp': depth_ts,
                'delta_ts_ms': None if dt_ms is None else round(dt_ms, 3),
            },
        }

    def _load_head_color_intrinsic(
        self,
        width: int,
        height: int,
        allow_gdk: bool = True,
    ) -> Dict[str, Any]:
        """加载头部彩色内参：JSON → GDK → HFOV 估计。"""
        names = (
            'intrinsic_head_color.json',
            'intrinsic_head_left_color.json',
            'intrinsic_head_rgb.json',
        )
        dirs = list(self.hand_intrinsic_search_dirs)
        if self.active_project and self.active_project.get('dir'):
            dirs.insert(0, self.active_project['dir'])
            dirs.insert(0, os.path.join(self.active_project['dir'], 'calib'))
        if HAS_D2C_PLANE:
            found = find_intrinsic_file(dirs, names)
            if found:
                return load_intrinsic_json(found, width, height)
        if allow_gdk and self.camera is not None:
            try:
                intr = self.camera.get_camera_intrinsic(
                    agibot_gdk.CameraType.kHeadColor)
                arr = list(getattr(intr, 'intrinsic', []) or [])
                if len(arr) >= 9:
                    fx, fy = float(arr[0]), float(arr[4])
                    cx, cy = float(arr[2]), float(arr[5])
                    if fx > 1 and fy > 1:
                        return parse_intrinsic(
                            {
                                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                                'source': 'gdk',
                            },
                            width, height,
                        )
            except Exception as e:
                print(f'⚠️ GDK 头部内参不可用: {e}')
        # 回退：头部 RGB 典型 HFOV ~70°（640x400 量级）
        import math
        hfov = math.radians(70.0)
        fx = width / (2.0 * math.tan(hfov / 2.0))
        return {
            'fx': float(fx), 'fy': float(fx),
            'cx': width / 2.0, 'cy': height / 2.0,
            'width': width, 'height': height,
            'source': 'hfov_estimate_head_70deg',
            'warning': '未找到头部彩色内参，使用 HFOV 估计',
        }

    def run_head_plane_metric(
        self,
        cols: int = 9,
        rows: int = 6,
        square_mm: float = 25.0,
        dilate_px: int = 12,
        z_min_mm: float = 200.0,
        z_max_mm: float = 3000.0,
        pixel_stride: int = 1,
        ransac_thresh_mm: float = 8.0,
        timeout_ms: float = 800.0,
        include_vis: bool = False,
    ) -> Dict[str, Any]:
        """头部 D2C：PnP 法向 vs 深度拟合。"""
        if not HAS_D2C_PLANE:
            return {'success': False, 'error': 'd2c_plane_metrics 未加载'}
        if self.camera is None:
            return {'success': False, 'error': '相机未初始化'}
        if not HAS_OPENCV:
            return {'success': False, 'error': '需要 OpenCV'}
        try:
            color_img = self.camera.get_latest_image(
                agibot_gdk.CameraType.kHeadColor, float(timeout_ms))
            depth_img = self.camera.get_latest_image(
                agibot_gdk.CameraType.kHeadDepth, float(timeout_ms))
        except Exception as e:
            return {'success': False, 'error': f'取帧失败: {e}'}
        if color_img is None or depth_img is None:
            return {'success': False, 'error': '头部彩色或深度为空'}

        color = self.decode_image_data(color_img, for_preview=False)
        depth_u16 = self._extract_depth_u16(depth_img, normalize_hand=False)
        if color is None or depth_u16 is None:
            return {'success': False, 'error': '头部彩色/深度解码失败'}
        if color.ndim == 2:
            color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
        elif color.shape[2] == 4:
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)

        ch, cw = color.shape[:2]
        try:
            intr = self._load_head_color_intrinsic(cw, ch, allow_gdk=True)
        except Exception as e:
            return {'success': False, 'error': str(e)}

        color_ts = getattr(color_img, 'timestamp', None)
        depth_ts = getattr(depth_img, 'timestamp', None)
        try:
            dt_ms = (
                abs(float(color_ts) - float(depth_ts)) * 1000.0
                if color_ts is not None and depth_ts is not None else None
            )
        except Exception:
            dt_ms = None

        result = evaluate_d2c_plane(
            color, depth_u16, intr,
            cols=int(cols), rows=int(rows), square_mm=float(square_mm),
            dilate_px=int(dilate_px),
            z_min_mm=float(z_min_mm), z_max_mm=float(z_max_mm),
            pixel_stride=int(pixel_stride),
            ransac_thresh_mm=float(ransac_thresh_mm),
            camera='head',
            delta_ts_ms=dt_ms,
            render_vis=bool(include_vis),
        )
        out = metrics_for_json(result)
        if include_vis and result.get('vis_bgr') is not None:
            ok, buf = cv2.imencode(
                '.jpg', result['vis_bgr'],
                [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                out['vis_jpeg_b64'] = base64.b64encode(buf.tobytes()).decode('ascii')
        return out

    def get_head_d2c_meta(self, timeout_ms: float = 120.0) -> Dict[str, Any]:
        """轻量 meta：分辨率与时间差（不合成大图）。"""
        if self.camera is None:
            return {'success': False, 'error': '相机未初始化'}
        try:
            color_img = self.camera.get_latest_image(
                agibot_gdk.CameraType.kHeadColor, float(timeout_ms))
            depth_img = self.camera.get_latest_image(
                agibot_gdk.CameraType.kHeadDepth, float(timeout_ms))
        except Exception as e:
            return {'success': False, 'error': str(e)}
        if color_img is None or depth_img is None:
            return {'success': False, 'error': '头部彩色或深度为空'}
        color_ts = getattr(color_img, 'timestamp', None)
        depth_ts = getattr(depth_img, 'timestamp', None)
        try:
            dt_ms = (
                abs(float(color_ts) - float(depth_ts)) * 1000.0
                if color_ts is not None and depth_ts is not None else None
            )
        except Exception:
            dt_ms = None
        depth_u16 = self._extract_depth_u16(depth_img, normalize_hand=False)
        valid_ratio = None
        if depth_u16 is not None:
            df = depth_u16.astype(np.float32)
            v = (
                (df >= self.depth_vis_valid_min_mm)
                & (df <= self.depth_vis_valid_max_mm)
            )
            valid_ratio = round(
                float(np.count_nonzero(v)) / float(v.size), 4)
        return {
            'success': True,
            'color_size': [
                int(getattr(color_img, 'width', 0) or 0),
                int(getattr(color_img, 'height', 0) or 0),
            ],
            'depth_size': [
                int(getattr(depth_img, 'width', 0) or 0),
                int(getattr(depth_img, 'height', 0) or 0),
            ],
            'color_timestamp': color_ts,
            'depth_timestamp': depth_ts,
            'delta_ts_ms': None if dt_ms is None else round(dt_ms, 3),
            'valid_depth_ratio': valid_ratio,
        }
    
    def get_camera_info(self, camera_type) -> str:
        """获取相机信息（短时缓存，避免每帧打 GDK）"""
        if self.camera is None:
            return "相机未初始化"

        now = time.time()
        cached = self._info_cache.get(camera_type)
        if cached and (now - cached[1]) < self._info_cache_ttl:
            return cached[0]

        try:
            shape = self.camera.get_image_shape(camera_type)
            fps = self.camera.get_image_fps(camera_type)
            text = f"尺寸: {shape[0]}x{shape[1]} | 帧率: {fps:.1f} FPS"
            self._info_cache[camera_type] = (text, now)
            return text
        except Exception as e:
            return f"获取信息失败: {e}"

    def prepare_preview_frame(self, decoded_image: np.ndarray) -> Optional[np.ndarray]:
        """缩小预览图，降低 WiFi/远程传输体积"""
        if decoded_image is None or not HAS_OPENCV:
            return decoded_image
        h, w = decoded_image.shape[:2]
        if w <= self.preview_max_width:
            return decoded_image
        scale = self.preview_max_width / float(w)
        nh = max(1, int(h * scale))
        return cv2.resize(
            decoded_image,
            (self.preview_max_width, nh),
            interpolation=cv2.INTER_AREA,
        )

    def encode_preview_jpeg(self, image_data) -> Optional[bytes]:
        """预览 JPEG（轻量，不做 OPTIMIZE）"""
        try:
            if not (HAS_OPENCV and isinstance(image_data, np.ndarray)):
                return None
            params = [cv2.IMWRITE_JPEG_QUALITY, int(self.preview_jpeg_quality)]
            ok, buffer = cv2.imencode('.jpg', image_data, params)
            if not ok:
                return None
            return buffer.tobytes()
        except Exception as e:
            print(f"预览 JPEG 编码失败: {e}")
            return None

    def get_preview_jpeg_bytes(self, camera_index: int) -> Optional[bytes]:
        """取单路预览 JPEG（供 /frame 低延迟推送）"""
        if self.camera is None or not (0 <= camera_index < len(self.camera_types)):
            return None
        now = time.monotonic()
        with self._preview_jpeg_lock:
            hit = self._preview_jpeg_cache.get(camera_index)
            if hit is not None:
                jpeg, ts, ok = hit
                ttl = self._preview_jpeg_cache_ttl if ok else self._preview_miss_cache_ttl
                if (now - ts) < ttl:
                    return jpeg if ok else None
        camera_type, _ = self.camera_types[camera_index]
        jpeg = None
        try:
            # GDK 在无帧时可能忽略短超时拖到 ~1s；用线程限时硬切，防止拖死 Flask
            image = self._get_latest_image_bounded(
                camera_type, float(self.preview_fetch_timeout_ms), hard_timeout_s=0.12
            )
            if image is not None:
                decoded = self.decode_image_data(image, for_preview=True)
                if decoded is not None:
                    preview = self.prepare_preview_frame(decoded)
                    jpeg = self.encode_preview_jpeg(preview)
        except Exception as e:
            print(f"预览取帧失败[{camera_index}]: {e}")
            jpeg = None
        with self._preview_jpeg_lock:
            self._preview_jpeg_cache[camera_index] = (jpeg, time.monotonic(), jpeg is not None)
        return jpeg

    def _get_latest_image_bounded(self, camera_type, timeout_ms: float, hard_timeout_s: float = 0.12):
        """get_latest_image 加硬超时，避免无帧时阻塞整页刷新。"""
        if self.camera is None:
            return None
        box: Dict[str, Any] = {'img': None, 'err': None}

        def _worker():
            try:
                box['img'] = self.camera.get_latest_image(camera_type, float(timeout_ms))
            except Exception as exc:
                box['err'] = exc

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        th.join(timeout=max(0.05, float(hard_timeout_s)))
        if th.is_alive():
            return None
        if box['err'] is not None:
            raise box['err']
        return box['img']

    def get_cached_image(self, camera_type):
        """获取缓存的图像"""
        current_time = time.time()
        
        if camera_type in self.image_cache:
            cached_image, cache_timestamp, image_timestamp = self.image_cache[camera_type]
            # 检查缓存是否有效且图像不是太旧
            if current_time - cache_timestamp < self.cache_timeout:
                return cached_image
        
        return None
    
    def cache_image(self, camera_type, image_b64, image_timestamp=None):
        """缓存图像"""
        if image_timestamp is None:
            image_timestamp = time.time()
        self.image_cache[camera_type] = (image_b64, time.time(), image_timestamp)
    
    def process_single_camera(self, camera_type, camera_name, index):
        """处理单个相机的图像（预览路径：缩小+低质 JPEG）"""
        try:
            # 获取图像（短超时，避免远程轮询被拖死）
            image = self.camera.get_latest_image(
                camera_type, float(self.preview_fetch_timeout_ms)
            )
            
            if image is not None:
                # 检查图像时间戳，避免显示历史帧
                image_timestamp = getattr(image, 'timestamp', None)
                if image_timestamp is None:
                    image_timestamp = time.time()
                
                # 检查是否是新的图像帧
                last_timestamp = self.last_image_timestamps.get(camera_type, 0)
                is_new_frame = image_timestamp > last_timestamp
                
                # 如果不是新帧，尝试使用缓存
                if not is_new_frame:
                    cached_image = self.get_cached_image(camera_type)
                    if cached_image:
                        info = self.get_camera_info(camera_type)
                        return {
                            'index': index,
                            'success': True,
                            'image': cached_image,
                            'info': info,
                            'available': True,
                            'cached': True
                        }
                
                # 处理新图像
                decoded_image = self.decode_image_data(image, for_preview=True)
                if decoded_image is not None:
                    preview = self.prepare_preview_frame(decoded_image)
                    image_b64 = self.image_to_base64(
                        preview, quality=int(self.preview_jpeg_quality)
                    )
                    if image_b64:
                        # 更新图像时间戳记录
                        self.last_image_timestamps[camera_type] = image_timestamp
                        
                        # 缓存新图像
                        self.cache_image(camera_type, image_b64, image_timestamp)
                        
                        info = self.get_camera_info(camera_type)
                        return {
                            'index': index,
                            'success': True,
                            'image': image_b64,
                            'info': info,
                            'available': True,
                            'cached': False
                        }
                    else:
                        return {
                            'index': index,
                            'success': False,
                            'error': '图像编码失败',
                            'available': False
                        }
                else:
                    return {
                        'index': index,
                        'success': False,
                        'error': '图像解码失败',
                        'available': False
                    }
            else:
                # 无法获取图像时，尝试使用缓存
                cached_image = self.get_cached_image(camera_type)
                if cached_image:
                    info = self.get_camera_info(camera_type)
                    return {
                        'index': index,
                        'success': True,
                        'image': cached_image,
                        'info': info,
                        'available': True,
                        'cached': True
                    }
                
                return {
                    'index': index,
                    'success': False,
                    'error': '无法获取图像',
                    'available': False
                }
        except Exception as e:
            error_msg = f'相机 {camera_name} 错误: {str(e)}'
            print(f"⚠️ {error_msg}")
            return {
                'index': index,
                'success': False,
                'error': error_msg,
                'available': False
            }
    
    def image_to_base64(self, image_data, quality=85) -> str:
        """将图像数据转换为base64字符串"""
        try:
            if HAS_OPENCV and isinstance(image_data, np.ndarray):
                # 预览不做 JPEG_OPTIMIZE（更慢）；落盘路径仍可传更高 quality
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
                _, buffer = cv2.imencode('.jpg', image_data, encode_params)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                return f"data:image/jpeg;base64,{img_base64}"
            elif isinstance(image_data, bytes):
                # 原始图像数据转base64
                img_base64 = base64.b64encode(image_data).decode('utf-8')
                return f"data:image/jpeg;base64,{img_base64}"
            else:
                return None
        except Exception as e:
            print(f"图像转base64失败: {e}")
            return None

    def _safe_camera_filename(self, camera_name: str) -> str:
        """生成可用于文件名的相机名"""
        name = re.sub(r'[^\w\u4e00-\u9fff\-]+', '_', camera_name).strip('_')
        return name or "camera"

    def _stereo_pair_indices(self) -> List[int]:
        """返回头部双目左/右在 camera_types 中的索引"""
        wanted = {
            agibot_gdk.CameraType.kHeadStereoLeft,
            agibot_gdk.CameraType.kHeadStereoRight,
        }
        indices = []
        for i, (camera_type, _) in enumerate(self.camera_types):
            if camera_type in wanted:
                indices.append(i)
        return indices

    def _indices_of_types(self, wanted) -> List[int]:
        """按给定 CameraType 集合/序列返回在 camera_types 中的索引（保持 wanted 顺序）。"""
        type_to_index = {ct: i for i, (ct, _) in enumerate(self.camera_types)}
        indices = []
        for ct in wanted:
            if ct in type_to_index:
                indices.append(type_to_index[ct])
        return indices

    def _capture_group_for_index(self, index: int) -> Dict[str, Any]:
        """
        根据用户点击的相机，决定本次工程落盘要采哪些路。
        - 头部双目任一目 → 左右双目
        - 左手/右手（RGBD）→ 深度+彩色；单目机型 → 仅彩色
        - 其他 → 仅该相机
        """
        if index < 0 or index >= len(self.camera_types):
            return {'success': False, 'error': '无效的相机索引', 'indices': [], 'label': ''}

        camera_type, camera_name = self.camera_types[index]
        head_left = agibot_gdk.CameraType.kHeadStereoLeft
        head_right = agibot_gdk.CameraType.kHeadStereoRight
        if camera_type in (head_left, head_right):
            indices = self._indices_of_types([head_left, head_right])
            return {
                'success': True,
                'kind': 'head_stereo',
                'indices': indices,
                'label': '头部双目',
                'trigger_index': index,
                'trigger_name': camera_name,
            }

        if self.hand_rgbd:
            hand_groups = [
                (
                    'hand_left',
                    '左手 RGBD',
                    [
                        agibot_gdk.CameraType.kHandLeftDepth,
                        agibot_gdk.CameraType.kHandLeftColor,
                    ],
                ),
                (
                    'hand_right',
                    '右手 RGBD',
                    [
                        agibot_gdk.CameraType.kHandRightDepth,
                        agibot_gdk.CameraType.kHandRightColor,
                    ],
                ),
            ]
        else:
            hand_groups = [
                (
                    'hand_left',
                    '左手彩色（单目）',
                    [
                        agibot_gdk.CameraType.kHandLeftColor,
                    ],
                ),
                (
                    'hand_right',
                    '右手彩色（单目）',
                    [
                        agibot_gdk.CameraType.kHandRightColor,
                    ],
                ),
            ]
        for kind, label, types in hand_groups:
            if camera_type in types:
                indices = self._indices_of_types(types)
                if not indices:
                    return {
                        'success': False,
                        'error': f'未配置{label}相机',
                        'indices': [],
                        'label': label,
                    }
                return {
                    'success': True,
                    'kind': kind,
                    'indices': indices,
                    'label': label,
                    'trigger_index': index,
                    'trigger_name': camera_name,
                }

        return {
            'success': True,
            'kind': 'single',
            'indices': [index],
            'label': camera_name,
            'trigger_index': index,
            'trigger_name': camera_name,
        }

    def _ensure_i2c_access(self) -> bool:
        """确保当前进程可访问 I2C（sg_standalone 需要）"""
        if os.access(I2C_DEV_PATH, os.R_OK | os.W_OK):
            return True
        password = os.environ.get("AGI_SUDO_PASSWORD", "1")
        try:
            result = subprocess.run(
                ["sudo", "-S", "chmod", "666", I2C_DEV_PATH],
                input=password + "\n",
                text=True,
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0 and os.access(I2C_DEV_PATH, os.R_OK | os.W_OK):
                print(f"✅ 已开放 I2C 权限: {I2C_DEV_PATH}")
                return True
            print(f"⚠️ 开放 I2C 权限失败: {result.stderr.strip()}")
        except Exception as e:
            print(f"⚠️ 开放 I2C 权限异常: {e}")
        return False

    def _load_sg_standalone(self):
        """加载同目录 / 家目录下的 sg_standalone.py"""
        for path in SG_STANDALONE_CANDIDATES:
            if not os.path.isfile(path):
                continue
            spec = importlib.util.spec_from_file_location("sg_standalone", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"📐 已加载内外参脚本: {path}")
            return module, path
        return None, None

    def save_head_stereo_camera_params(self, output_dir: str) -> Dict[str, Any]:
        """
        使用 sg_standalone.py 读取头部双目内外参并落盘
        （支持 fisheye / pinhole，输出与生产流水线 JSON 命名一致）
        """
        if not self._ensure_i2c_access():
            return {
                'success': False,
                'error': f'无权限访问 {I2C_DEV_PATH}，请将用户加入 i2c 组或设置 AGI_SUDO_PASSWORD',
            }

        sg, sg_path = self._load_sg_standalone()
        if sg is None:
            return {
                'success': False,
                'error': '未找到 sg_standalone.py，请放到与 camera_webview.py 同目录或 ~/sg_standalone.py',
            }

        try:
            os.makedirs(output_dir, exist_ok=True)
            fw = sg.read_firmware(SG_I2C_BUS, SG_LEFT_ADDR) or ""
            bufs = sg.read_three_blocks(SG_I2C_BUS, SG_LEFT_ADDR, SG_DES_ADDR)
            if bufs is None:
                return {'success': False, 'error': 'sg_standalone 读取 I2C 三块数据失败'}

            left, right, ext, sn, meta = sg.decode_stereo(bufs)
            if not sg.validate_intrinsic(left):
                return {'success': False, 'error': '左目内参无效（Fx/Fy/Cx/Cy）'}
            if not sg.validate_intrinsic(right):
                return {'success': False, 'error': '右目内参无效（Fx/Fy/Cx/Cy）'}

            sg.write_json_files(output_dir, left, right, ext, sn, fw)
            saved_files = [
                os.path.join(output_dir, "intrinsic_head_left_stereo.json"),
                os.path.join(output_dir, "intrinsic_head_right_stereo.json"),
                os.path.join(output_dir, "extrinsic_head_right_T_left_stereo.json"),
            ]
            # 额外写一份完整汇总，便于核对
            summary_path = os.path.join(output_dir, "stereo_params_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sn": sn,
                        "firmware": fw,
                        "model": meta.get("model") if isinstance(meta, dict) else None,
                        "meta": meta,
                        "left_intrinsic": left,
                        "right_intrinsic": right,
                        "extrinsic_mm": ext,
                        "source_script": sg_path,
                        "i2c": {
                            "bus": SG_I2C_BUS,
                            "left_addr": hex(SG_LEFT_ADDR),
                            "right_addr": hex(SG_RIGHT_ADDR),
                            "des_addr": hex(SG_DES_ADDR),
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            saved_files.append(summary_path)

            print(
                f"📐 已落盘双目内外参 model={meta.get('model') if isinstance(meta, dict) else meta} "
                f"SN={sn}: {saved_files}"
            )
            return {
                'success': True,
                'files': saved_files,
                'sn': sn,
                'firmware': fw,
                'model': meta.get("model") if isinstance(meta, dict) else None,
                'save_dir': output_dir,
            }
        except Exception as e:
            return {'success': False, 'error': f'sg_standalone 读取/写入失败: {e}'}

    def capture_camera_to_disk(
        self,
        index: int,
        timestamp: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """采集指定相机当前帧并落盘；可传入统一 timestamp / output_dir 方便双目配对"""
        if self.camera is None:
            return {'success': False, 'error': '相机未初始化'}
        if index < 0 or index >= len(self.camera_types):
            return {'success': False, 'error': '无效的相机索引'}

        camera_type, camera_name = self.camera_types[index]
        try:
            image = self.camera.get_latest_image(camera_type, 1000.0)
            if image is None:
                return {'success': False, 'error': f'无法获取 {camera_name} 图像'}

            decoded_image = self.decode_image_data(image)
            if decoded_image is None:
                return {'success': False, 'error': f'无法解码 {camera_name} 图像'}

            ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            out_dir = output_dir or self.save_dir
            os.makedirs(out_dir, exist_ok=True)
            safe_name = self._safe_camera_filename(camera_name)
            saved_files = []

            # 可视化/彩色图落盘为 JPEG
            if HAS_OPENCV and isinstance(decoded_image, np.ndarray):
                jpg_path = os.path.join(out_dir, f"{safe_name}_{ts}.jpg")
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                if not cv2.imwrite(jpg_path, decoded_image, encode_params):
                    return {'success': False, 'error': 'JPEG 写入失败'}
                saved_files.append(jpg_path)
            elif isinstance(decoded_image, (bytes, bytearray)):
                jpg_path = os.path.join(out_dir, f"{safe_name}_{ts}.jpg")
                with open(jpg_path, 'wb') as f:
                    f.write(decoded_image)
                saved_files.append(jpg_path)
            else:
                return {'success': False, 'error': '不支持的图像数据类型'}

            # 深度等未压缩原始数据额外存 npy，便于后续处理
            encoding = getattr(image, 'encoding', None)
            color_format = getattr(image, 'color_format', None)
            if (encoding == agibot_gdk.Encoding.UNCOMPRESSED and
                    hasattr(image, 'data') and image.data is not None):
                try:
                    if color_format == agibot_gdk.ColorFormat.RS2_FORMAT_Z16:
                        raw = np.frombuffer(image.data, dtype=np.uint16)
                        depth_arr = self._reshape_depth_u16(
                            raw, int(image.width or 0), int(image.height or 0))
                        depth_arr = self._prepare_hand_depth_u16(depth_arr)
                        npy_path = os.path.join(out_dir, f"{safe_name}_{ts}_depth_u16.npy")
                        np.save(npy_path, depth_arr)
                        saved_files.append(npy_path)
                    elif color_format == agibot_gdk.ColorFormat.GRAY16:
                        raw = np.frombuffer(image.data, dtype=np.uint16).reshape(
                            (image.height, image.width))
                        npy_path = os.path.join(out_dir, f"{safe_name}_{ts}_gray16.npy")
                        np.save(npy_path, raw)
                        saved_files.append(npy_path)
                except Exception as e:
                    print(f"⚠️ 原始深度落盘失败: {e}")

            print(f"💾 已采集 {camera_name}: {saved_files}")
            return {
                'success': True,
                'camera_name': camera_name,
                'index': index,
                'files': saved_files,
                'timestamp': ts,
                'save_dir': out_dir,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _hand_rgbd_pair_for_index(self, index: int):
        """根据相机索引返回 (depth_type, color_type, side, depth_name) 或 None。

        深度卡或彩色卡均可：左手/右手深度、左手/右手彩色。
        """
        if index < 0 or index >= len(self.camera_types):
            return None
        cam_type, name = self.camera_types[index]
        by_depth = {
            agibot_gdk.CameraType.kHandLeftDepth: (
                agibot_gdk.CameraType.kHandLeftDepth,
                agibot_gdk.CameraType.kHandLeftColor, 'left', '左手深度相机'),
            agibot_gdk.CameraType.kHandRightDepth: (
                agibot_gdk.CameraType.kHandRightDepth,
                agibot_gdk.CameraType.kHandRightColor, 'right', '右手深度相机'),
        }
        by_color = {
            agibot_gdk.CameraType.kHandLeftColor: (
                agibot_gdk.CameraType.kHandLeftDepth,
                agibot_gdk.CameraType.kHandLeftColor, 'left', '左手深度相机'),
            agibot_gdk.CameraType.kHandRightColor: (
                agibot_gdk.CameraType.kHandRightDepth,
                agibot_gdk.CameraType.kHandRightColor, 'right', '右手深度相机'),
        }
        if cam_type in by_depth:
            return by_depth[cam_type]
        if cam_type in by_color:
            return by_color[cam_type]
        return None

    def _head_rgbd_pair_for_index(self, index: int):
        """头部彩色/深度索引 → (depth_type, color_type, 'head', depth_name)。"""
        if index < 0 or index >= len(self.camera_types):
            return None
        cam_type, _name = self.camera_types[index]
        if cam_type in (
            agibot_gdk.CameraType.kHeadDepth,
            agibot_gdk.CameraType.kHeadColor,
        ):
            return (
                agibot_gdk.CameraType.kHeadDepth,
                agibot_gdk.CameraType.kHeadColor,
                'head',
                '头部深度相机',
            )
        return None

    def _rgbd_pair_for_index(self, index: int):
        """手部或头部 RGBD 配对。"""
        return (
            self._hand_rgbd_pair_for_index(index)
            or self._head_rgbd_pair_for_index(index)
        )

    def build_hand_d2c_overlay(
        self,
        index: int,
        alpha: float = 0.45,
        z_min_mm: float = 200.0,
        z_max_mm: float = 3000.0,
        mode: str = 'overlay',
        edges: bool = False,
        timeout_ms: float = 150.0,
    ) -> Dict[str, Any]:
        """手部彩色 + D2C 深度叠加（深度已对齐到 RGB 画布）。"""
        if self.camera is None:
            return {'success': False, 'error': '相机未初始化'}
        if not HAS_OPENCV:
            return {'success': False, 'error': '需要 OpenCV'}
        pair = self._hand_rgbd_pair_for_index(index)
        if pair is None:
            return {'success': False, 'error': '请选择左手/右手彩色或深度相机'}
        depth_type, color_type, side, depth_name = pair

        alpha = float(np.clip(alpha, 0.0, 1.0))
        mode = (mode or 'overlay').strip().lower()
        if mode not in ('overlay', 'side', 'color'):
            mode = 'overlay'

        try:
            color_img = self.camera.get_latest_image(color_type, float(timeout_ms))
            depth_img = self.camera.get_latest_image(depth_type, float(timeout_ms))
        except Exception as e:
            return {'success': False, 'error': f'取帧失败: {e}'}
        if color_img is None or depth_img is None:
            return {'success': False, 'error': '手部彩色或深度为空'}

        color = self._decode_color_bgr(color_img)
        # D2C 深度保持原生 1280x1056，勿做手部 1280x960 规整
        depth_u16 = self._extract_depth_u16(depth_img, normalize_hand=False)
        if color is None or depth_u16 is None:
            return {'success': False, 'error': '手部彩色/深度解码失败'}
        if color.ndim == 2:
            color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
        elif color.shape[2] == 4:
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)

        ch, cw = color.shape[:2]
        dh, dw = depth_u16.shape[:2]
        size_matched = (dh, dw) == (ch, cw)
        if not size_matched:
            depth_u16 = cv2.resize(
                depth_u16, (cw, ch), interpolation=cv2.INTER_NEAREST)

        depth_vis, valid = self._colorize_depth_u16(
            depth_u16,
            for_preview=False,
            z_min_mm=z_min_mm,
            z_max_mm=z_max_mm,
            draw_label=False,
            return_valid=True,
        )

        if mode == 'color':
            out = color.copy()
        else:
            blend = color.copy()
            if np.any(valid):
                a = alpha
                overlay_region = (
                    color[valid].astype(np.float32) * (1.0 - a)
                    + depth_vis[valid].astype(np.float32) * a
                )
                blend[valid] = np.clip(overlay_region, 0, 255).astype(np.uint8)
            if edges and np.any(valid):
                edge = cv2.Canny((valid.astype(np.uint8) * 255), 50, 150)
                blend[edge > 0] = (0, 255, 255)
            label = f'Hand-{side} D2C a={alpha:.2f} z={int(z_min_mm)}-{int(z_max_mm)}mm'
            cv2.putText(
                blend, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
            if mode == 'side':
                gap = 8
                canvas = np.zeros((ch, cw * 3 + gap * 2, 3), dtype=np.uint8)
                canvas[:, 0:cw] = color
                canvas[:, cw + gap:cw * 2 + gap] = depth_vis
                canvas[:, cw * 2 + gap * 2:] = blend
                for i, title in enumerate(('Color', 'Depth', 'D2C')):
                    x0 = i * (cw + gap) + 10
                    cv2.putText(
                        canvas, title, (x0, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
                out = canvas
            else:
                out = blend

        preview = self.prepare_preview_frame(out)
        jpeg = self.encode_preview_jpeg(preview if preview is not None else out)
        if not jpeg:
            return {'success': False, 'error': 'JPEG 编码失败'}

        color_ts = getattr(color_img, 'timestamp', None)
        depth_ts = getattr(depth_img, 'timestamp', None)
        try:
            dt_ms = (
                abs(float(color_ts) - float(depth_ts)) * 1000.0
                if color_ts is not None and depth_ts is not None else None
            )
        except Exception:
            dt_ms = None
        valid_ratio = float(np.count_nonzero(valid)) / float(valid.size)
        return {
            'success': True,
            'jpeg': jpeg,
            'meta': {
                'side': side,
                'depth_name': depth_name,
                'color_size': [int(cw), int(ch)],
                'depth_native_size': [int(dw), int(dh)],
                'aligned_size': [int(cw), int(ch)],
                'size_matched': bool(size_matched),
                'preview_size': [
                    int((preview if preview is not None else out).shape[1]),
                    int((preview if preview is not None else out).shape[0]),
                ],
                'valid_depth_ratio': valid_ratio,
                'delta_ts_ms': dt_ms,
                'alpha': alpha,
                'z_min_mm': float(z_min_mm),
                'z_max_mm': float(z_max_mm),
                'mode': mode,
            },
        }

    def probe_hand_depth_at(
        self,
        index: int,
        nx: float,
        ny: float,
        timeout_ms: float = 150.0,
    ) -> Dict[str, Any]:
        """在彩色图归一化坐标 (nx,ny)∈[0,1] 处读取对齐深度(mm)。"""
        if self.camera is None:
            return {'success': False, 'error': '相机未初始化'}
        pair = self._hand_rgbd_pair_for_index(index)
        if pair is None:
            return {'success': False, 'error': '请选择左手/右手彩色或深度相机'}
        depth_type, color_type, side, depth_name = pair
        try:
            nx = float(nx)
            ny = float(ny)
        except Exception:
            return {'success': False, 'error': '坐标无效'}
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            return {'success': False, 'error': '坐标需在 [0,1]'}

        try:
            color_img = self.camera.get_latest_image(color_type, float(timeout_ms))
            depth_img = self.camera.get_latest_image(depth_type, float(timeout_ms))
        except Exception as e:
            return {'success': False, 'error': f'取帧失败: {e}'}
        if color_img is None or depth_img is None:
            return {'success': False, 'error': '手部彩色或深度为空'}

        color = self._decode_color_bgr(color_img)
        depth_u16 = self._extract_depth_u16(depth_img, normalize_hand=False)
        if color is None or depth_u16 is None:
            return {'success': False, 'error': '解码失败'}
        ch, cw = color.shape[:2]
        dh, dw = depth_u16.shape[:2]
        if (dh, dw) != (ch, cw) and HAS_OPENCV:
            depth_u16 = cv2.resize(
                depth_u16, (cw, ch), interpolation=cv2.INTER_NEAREST)
            dh, dw = ch, cw

        u = int(np.clip(round(nx * (cw - 1)), 0, cw - 1))
        v = int(np.clip(round(ny * (ch - 1)), 0, ch - 1))
        z_mm = int(depth_u16[v, u]) if dh == ch and dw == cw else 0
        # 邻域中位数，减少单点噪声
        x0, x1 = max(0, u - 2), min(cw, u + 3)
        y0, y1 = max(0, v - 2), min(ch, v + 3)
        patch = depth_u16[y0:y1, x0:x1]
        valid = patch[patch > 0]
        z_med = int(np.median(valid)) if valid.size else 0
        bgr = color[v, u].tolist() if color.ndim == 3 else [int(color[v, u])] * 3
        return {
            'success': True,
            'side': side,
            'depth_name': depth_name,
            'u': u,
            'v': v,
            'nx': nx,
            'ny': ny,
            'color_size': [int(cw), int(ch)],
            'depth_size': [int(dw), int(dh)],
            'z_mm': z_mm,
            'z_mm_median5': z_med,
            'valid': bool(z_mm > 0),
            'bgr': bgr,
            'rgb': [int(bgr[2]), int(bgr[1]), int(bgr[0])] if len(bgr) >= 3 else bgr,
        }

    def probe_head_depth_at(
        self,
        nx: float,
        ny: float,
        timeout_ms: float = 150.0,
    ) -> Dict[str, Any]:
        """在头部彩色图归一化坐标 (nx,ny)∈[0,1] 处读取对齐深度(mm)。"""
        if self.camera is None:
            return {'success': False, 'error': '相机未初始化'}
        try:
            nx = float(nx)
            ny = float(ny)
        except Exception:
            return {'success': False, 'error': '坐标无效'}
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            return {'success': False, 'error': '坐标需在 [0,1]'}

        try:
            color_img = self.camera.get_latest_image(
                agibot_gdk.CameraType.kHeadColor, float(timeout_ms))
            depth_img = self.camera.get_latest_image(
                agibot_gdk.CameraType.kHeadDepth, float(timeout_ms))
        except Exception as e:
            return {'success': False, 'error': f'取帧失败: {e}'}
        if color_img is None or depth_img is None:
            return {'success': False, 'error': '头部彩色或深度为空'}

        color = self._decode_color_bgr(color_img)
        depth_u16 = self._extract_depth_u16(depth_img, normalize_hand=False)
        if color is None or depth_u16 is None:
            return {'success': False, 'error': '解码失败'}
        ch, cw = color.shape[:2]
        dh, dw = depth_u16.shape[:2]
        if (dh, dw) != (ch, cw) and HAS_OPENCV:
            depth_u16 = cv2.resize(
                depth_u16, (cw, ch), interpolation=cv2.INTER_NEAREST)
            dh, dw = ch, cw

        u = int(np.clip(round(nx * (cw - 1)), 0, cw - 1))
        v = int(np.clip(round(ny * (ch - 1)), 0, ch - 1))
        z_mm = int(depth_u16[v, u]) if dh == ch and dw == cw else 0
        x0, x1 = max(0, u - 2), min(cw, u + 3)
        y0, y1 = max(0, v - 2), min(ch, v + 3)
        patch = depth_u16[y0:y1, x0:x1]
        valid = patch[patch > 0]
        z_med = int(np.median(valid)) if valid.size else 0
        bgr = color[v, u].tolist() if color.ndim == 3 else [int(color[v, u])] * 3
        return {
            'success': True,
            'side': 'head',
            'depth_name': '头部深度相机',
            'u': u,
            'v': v,
            'nx': nx,
            'ny': ny,
            'color_size': [int(cw), int(ch)],
            'depth_size': [int(dw), int(dh)],
            'z_mm': z_mm,
            'z_mm_median5': z_med,
            'valid': bool(z_mm > 0),
            'bgr': bgr,
            'rgb': [int(bgr[2]), int(bgr[1]), int(bgr[0])] if len(bgr) >= 3 else bgr,
        }

    def _hand_intrinsic_dirs(self) -> List[str]:
        dirs = list(self.hand_intrinsic_search_dirs)
        if self.active_project and self.active_project.get('dir'):
            dirs.insert(0, self.active_project['dir'])
        return dirs

    @staticmethod
    def _scale_hand_intrinsics_to_size(
        fx: float, fy: float, cx: float, cy: float,
        src_w: int, src_h: int, dst_w: int, dst_h: int,
    ) -> Dict[str, float]:
        """将标定分辨率缩放到实际深度图尺寸。同宽仅垂直居中裁时只改 cy。"""
        src_w, src_h = int(src_w), int(src_h)
        dst_w, dst_h = int(dst_w), int(dst_h)
        if src_w <= 0 or src_h <= 0:
            return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        if src_w == dst_w and src_h == dst_h:
            return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        if src_w == dst_w and src_h > dst_h:
            y0 = (src_h - dst_h) / 2.0
            return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy - y0}
        sx = dst_w / float(src_w)
        sy = dst_h / float(src_h)
        return {
            'fx': fx * sx, 'fy': fy * sy,
            'cx': cx * sx, 'cy': cy * sy,
        }

    def _intr_from_params_hand_rgbd(
        self, path: str, width: int, height: int, which: str = 'tof',
    ) -> Optional[Dict[str, Any]]:
        """从 params_hand_*_rgbd.json 取内参。

        which='tof' → Intr_TOF @ tof_size(640x480)
        which='rgb' → Intr_RGB @ rgb_size(1280x1056)，用于 D2C 深度点云
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        intrs = data.get('Intrs') or {}
        which = (which or 'tof').lower()
        if which == 'rgb':
            K = intrs.get('Intr_RGB') or []
            size = intrs.get('rgb_size') or [1280, 1056]
            dist = intrs.get('distCoeffs_RGB') or []
            key_name = 'Intr_RGB'
        else:
            K = intrs.get('Intr_TOF') or []
            size = intrs.get('tof_size') or [640, 480]
            dist = intrs.get('distCoeffs_TOF') or []
            key_name = 'Intr_TOF'
        if len(K) < 6:
            return None
        src_w, src_h = int(size[0]), int(size[1])
        fx, fy = float(K[0]), float(K[4])
        cx, cy = float(K[2]), float(K[5])
        scaled = self._scale_hand_intrinsics_to_size(
            fx, fy, cx, cy, src_w, src_h, width, height)
        meta = {
            'source': f'file:{path}:{key_name}',
            'sn': data.get('SN'),
            'model': key_name,
            'fx': scaled['fx'], 'fy': scaled['fy'],
            'cx': scaled['cx'], 'cy': scaled['cy'],
            'calib_width': src_w, 'calib_height': src_h,
            'distortion': {},
            'extrinsic_tof_T_rgb': {
                'R_flat': intrs.get('RDtoRGB'),
                'T_m': intrs.get('TDtoRGB'),
                'unit': 'meter',
            },
        }
        if len(dist) >= 5:
            meta['distortion'] = {
                'k1': float(dist[0]), 'k2': float(dist[1]),
                'p1': float(dist[2]), 'p2': float(dist[3]),
                'k3': float(dist[4]),
            }
        return meta

    def _load_hand_pc_intrinsic(
        self,
        side: str,
        width: int,
        height: int,
        mode: str = 'd2c',
        allow_gdk: bool = True,
    ) -> Dict[str, Any]:
        """点云内参：d2c 用 Intr_RGB；tof 用 Intr_TOF。"""
        mode = (mode or 'd2c').lower()
        if mode != 'd2c':
            return self._load_hand_depth_intrinsic(
                side, width, height, allow_gdk=allow_gdk)

        import math
        cache_key = ('pc_rgb', side, int(width), int(height))
        cached = self._hand_intr_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        meta = {
            'source': None, 'width': width, 'height': height,
            'model': 'Intr_RGB', 'pc_mode': 'd2c',
        }
        dirs = self._hand_intrinsic_dirs()
        candidates = []
        for d in dirs:
            candidates.append(os.path.join(d, f'params_hand_{side}_rgbd.json'))
            candidates.append(os.path.join(d, f'intrinsic_hand_{side}_depth.json'))

        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                if os.path.basename(path).startswith('params_hand_'):
                    loaded = self._intr_from_params_hand_rgbd(
                        path, width, height, which='rgb')
                    if loaded is None:
                        continue
                    meta.update(loaded)
                    meta['pc_mode'] = 'd2c'
                    print(
                        f'📐 手部D2C点云内参[{side}] <- {path} Intr_RGB '
                        f'fx={meta["fx"]:.2f} fy={meta["fy"]:.2f} '
                        f'cx={meta["cx"]:.2f} cy={meta["cy"]:.2f} @ {width}x{height}',
                        flush=True,
                    )
                    self._hand_intr_cache[cache_key] = meta
                    return dict(meta)

                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # intrinsic_hand_*_depth.json 内嵌 Intr_RGB_raw @ rgb_size_raw
                K = data.get('Intr_RGB_raw') or []
                if len(K) < 6:
                    continue
                rgb_size = data.get('rgb_size_raw') or [1280, 1056]
                src_w, src_h = int(rgb_size[0]), int(rgb_size[1])
                fx, fy = float(K[0]), float(K[4])
                cx, cy = float(K[2]), float(K[5])
                scaled = self._scale_hand_intrinsics_to_size(
                    fx, fy, cx, cy, src_w, src_h, width, height)
                meta.update({
                    'source': f'file:{path}:Intr_RGB_raw',
                    'sn': data.get('SN'),
                    'model': 'Intr_RGB',
                    'pc_mode': 'd2c',
                    'fx': scaled['fx'], 'fy': scaled['fy'],
                    'cx': scaled['cx'], 'cy': scaled['cy'],
                    'calib_width': src_w, 'calib_height': src_h,
                })
                print(
                    f'📐 手部D2C点云内参[{side}] <- {path} Intr_RGB '
                    f'fx={meta["fx"]:.2f} fy={meta["fy"]:.2f} '
                    f'cx={meta["cx"]:.2f} cy={meta["cy"]:.2f} @ {width}x{height}',
                    flush=True,
                )
                self._hand_intr_cache[cache_key] = meta
                return dict(meta)
            except Exception as e:
                print(f'⚠️ 读取 RGB 内参失败 {path}: {e}')

        # 回退：按 RGB 典型 HFOV(~105°) 估计，避免误用 TOF 内参造成锥形
        hfov = math.radians(105.0)
        fx = width / (2.0 * math.tan(hfov / 2.0))
        meta.update({
            'source': 'hfov_estimate_rgb_105deg',
            'fx': float(fx), 'fy': float(fx),
            'cx': width / 2.0, 'cy': height / 2.0,
            'warning': '未找到 Intr_RGB，使用 RGB FOV 估计；请检查 camera_intrinsics',
        })
        self._hand_intr_cache[cache_key] = meta
        return dict(meta)

    def _load_hand_depth_intrinsic(
        self,
        side: str,
        width: int,
        height: int,
        allow_gdk: bool = True,
    ) -> Dict[str, Any]:
        """加载手部深度内参：JSON / params_hand_* → GDK(可跳过) → FOV 估计。"""
        import math
        cache_key = (side, int(width), int(height), bool(allow_gdk))
        cached = self._hand_intr_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        meta = {'source': None, 'width': width, 'height': height}
        dirs = self._hand_intrinsic_dirs()
        candidates = []
        for d in dirs:
            candidates.append(os.path.join(d, f'intrinsic_hand_{side}_depth.json'))
            candidates.append(os.path.join(d, f'params_hand_{side}_rgbd.json'))

        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                if os.path.basename(path).startswith('params_hand_'):
                    loaded = self._intr_from_params_hand_rgbd(path, width, height)
                    if loaded is None:
                        continue
                    meta.update(loaded)
                    print(
                        f'📐 手部点云内参[{side}] <- {path} '
                        f'fx={meta["fx"]:.2f} fy={meta["fy"]:.2f} '
                        f'cx={meta["cx"]:.2f} cy={meta["cy"]:.2f} @ {width}x{height}',
                        flush=True,
                    )
                    self._hand_intr_cache[cache_key] = meta
                    return dict(meta)

                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                fx, fy = float(data['Fx']), float(data['Fy'])
                cx, cy = float(data['Cx']), float(data['Cy'])
                src_w = int(data.get('width') or width)
                src_h = int(data.get('height') or height)
                scaled = self._scale_hand_intrinsics_to_size(
                    fx, fy, cx, cy, src_w, src_h, width, height)
                meta.update({
                    'source': f'file:{path}',
                    'sn': data.get('SN'),
                    'fx': scaled['fx'], 'fy': scaled['fy'],
                    'cx': scaled['cx'], 'cy': scaled['cy'],
                    'calib_width': src_w, 'calib_height': src_h,
                    'distortion': {
                        k: data.get(k) for k in ('k1', 'k2', 'k3', 'p1', 'p2') if k in data
                    },
                })
                if data.get('RDtoRGB_flat') or data.get('TDtoRGB'):
                    meta['extrinsic_tof_T_rgb'] = {
                        'R_flat': data.get('RDtoRGB_flat'),
                        'T_m': data.get('TDtoRGB'),
                        'unit': 'meter',
                    }
                else:
                    # companion extrinsic file
                    extr_path = os.path.join(
                        os.path.dirname(path),
                        f'extrinsic_hand_{side}_tof_T_rgb.json',
                    )
                    if os.path.isfile(extr_path):
                        with open(extr_path, 'r', encoding='utf-8') as ef:
                            extr = json.load(ef)
                        meta['extrinsic_tof_T_rgb'] = {
                            'R_flat': extr.get('R_flat') or (
                                [c for row in extr.get('R', []) for c in row]
                                if extr.get('R') else None
                            ),
                            'T_m': extr.get('T_m') or extr.get('T'),
                            'unit': extr.get('unit', 'meter'),
                            'source': extr_path,
                        }
                print(
                    f'📐 手部点云内参[{side}] <- {path} '
                    f'fx={meta["fx"]:.2f} fy={meta["fy"]:.2f} '
                    f'cx={meta["cx"]:.2f} cy={meta["cy"]:.2f} @ {width}x{height}',
                    flush=True,
                )
                self._hand_intr_cache[cache_key] = meta
                return dict(meta)
            except Exception as e:
                print(f'⚠️ 读取内参失败 {path}: {e}')

        # GDK 配置服务常超时；同一侧只尝试一次，避免实时预览卡死
        if allow_gdk and side not in self._hand_intr_gdk_tried:
            self._hand_intr_gdk_tried.add(side)
            try:
                ct = (agibot_gdk.CameraType.kHandLeftDepth if side == 'left'
                      else agibot_gdk.CameraType.kHandRightDepth)
                intr = self.camera.get_camera_intrinsic(ct)
                arr = list(getattr(intr, 'intrinsic', []) or [])
                if len(arr) >= 9:
                    fx, fy, cx, cy = float(arr[0]), float(arr[4]), float(arr[2]), float(arr[5])
                    if fx > 1 and fy > 1:
                        meta.update({'source': 'gdk', 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy})
                        self._hand_intr_cache[cache_key] = meta
                        return dict(meta)
            except Exception as e:
                print(f'⚠️ GDK 内参不可用: {e}')

        hfov = math.radians(float(self.hand_depth_hfov_deg))
        fx = width / (2.0 * math.tan(hfov / 2.0))
        meta.update({
            'source': f'hfov_estimate_{self.hand_depth_hfov_deg}deg',
            'fx': float(fx), 'fy': float(fx),
            'cx': width / 2.0, 'cy': height / 2.0,
            'warning': '未找到工厂内参，使用水平 FOV 估计；点云尺度/畸变可能不准',
        })
        self._hand_intr_cache[cache_key] = meta
        return dict(meta)

    def _decode_color_bgr(self, image) -> Optional[np.ndarray]:
        """解码彩色帧为 BGR uint8。"""
        if image is None:
            return None
        try:
            enc = getattr(image, 'encoding', None)
            if enc == agibot_gdk.Encoding.JPEG and HAS_OPENCV:
                buf = np.frombuffer(image.data, dtype=np.uint8)
                return cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return self.decode_image_data(image, for_preview=False)
        except Exception as e:
            print(f'⚠️ 彩色解码失败: {e}')
            return None

    def depth_to_pointcloud(
        self,
        depth_u16: np.ndarray,
        color_bgr: Optional[np.ndarray],
        intr: Dict[str, Any],
        pixel_stride: int = 1,
        z_min_mm: Optional[float] = None,
        z_max_mm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Z16 深度(mm) + 可选彩色 → xyz(米) / rgb。同尺寸时按像素对齐着色。

        默认不做距离门限：只要深度值 > 0（有数据）就出点。
        若显式传入 z_min_mm / z_max_mm 才额外按范围过滤。
        """
        if depth_u16 is None or depth_u16.ndim != 2:
            raise ValueError('depth_u16 必须是 HxW uint16')
        h, w = depth_u16.shape
        stride = max(1, int(pixel_stride))
        fx, fy = float(intr['fx']), float(intr['fy'])
        cx, cy = float(intr['cx']), float(intr['cy'])

        vs = np.arange(0, h, stride, dtype=np.int32)
        us = np.arange(0, w, stride, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs)
        z_mm = depth_u16[vv, uu].astype(np.float32)
        # 有数据即保留；0 视为无回波/空像素
        valid = z_mm > 0
        if z_min_mm is not None:
            valid &= z_mm >= float(z_min_mm)
        if z_max_mm is not None:
            valid &= z_mm <= float(z_max_mm)
        if not np.any(valid):
            return {
                'xyz': np.zeros((0, 3), dtype=np.float32),
                'rgb': np.zeros((0, 3), dtype=np.uint8),
                'count': 0,
            }

        uu_v = uu[valid].astype(np.float32)
        vv_v = vv[valid].astype(np.float32)
        z = z_mm[valid] / 1000.0
        x = (uu_v - cx) * z / fx
        y = (vv_v - cy) * z / fy
        xyz = np.stack([x, y, z], axis=1).astype(np.float32)

        rgb = None
        ui = uu[valid]
        vi = vv[valid]
        if color_bgr is not None and isinstance(color_bgr, np.ndarray):
            ch, cw = color_bgr.shape[:2]
            src = color_bgr
            if ch != h or cw != w:
                if HAS_OPENCV:
                    src = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    src = None
            if src is not None:
                bgr = src[vi, ui]
                rgb = bgr[:, ::-1].copy()
        if rgb is None:
            rgb = np.full((xyz.shape[0], 3), 200, dtype=np.uint8)
        return {'xyz': xyz, 'rgb': rgb, 'count': int(xyz.shape[0])}

    def _write_ply_xyzrgb(self, path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
        n = int(xyz.shape[0])
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode('ascii')
        verts = np.empty(n, dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
        ])
        verts['x'] = xyz[:, 0]
        verts['y'] = xyz[:, 1]
        verts['z'] = xyz[:, 2]
        verts['r'] = rgb[:, 0]
        verts['g'] = rgb[:, 1]
        verts['b'] = rgb[:, 2]
        with open(path, 'wb') as f:
            f.write(header)
            f.write(verts.tobytes())

    def build_hand_pointcloud(
        self,
        index: int,
        pixel_stride: int = 1,
        allow_gdk_intrinsic: bool = True,
    ) -> Dict[str, Any]:
        """构建一手 RGBD 点云（不落盘），供预览/录制复用。"""
        if self.camera is None:
            return {'success': False, 'error': '相机未初始化'}
        pair = self._hand_rgbd_pair_for_index(index)
        if pair is None:
            return {'success': False, 'error': '请选择左手/右手深度相机'}

        depth_type, color_type, side, depth_name = pair
        try:
            depth_img = self.camera.get_latest_image(depth_type, 200.0)
            if depth_img is None:
                return {'success': False, 'error': f'无法获取{depth_name}'}
            if getattr(depth_img, 'color_format', None) != agibot_gdk.ColorFormat.RS2_FORMAT_Z16:
                return {'success': False, 'error': '深度格式不是 Z16'}

            depth = self._reshape_depth_u16(
                np.frombuffer(depth_img.data, dtype=np.uint16),
                int(depth_img.width or 0),
                int(depth_img.height or 0),
            )
            pc_mode = str(getattr(self, 'hand_pc_mode', 'd2c') or 'd2c').lower()
            if pc_mode == 'tof':
                # 原生 TOF：缩到 Intr_TOF 标定分辨率
                pc_w, pc_h = self.hand_pc_depth_size
                if depth.shape[1] != pc_w or depth.shape[0] != pc_h:
                    if HAS_OPENCV:
                        depth = cv2.resize(
                            depth, (pc_w, pc_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        sh = max(1, depth.shape[0] // pc_h)
                        sw = max(1, depth.shape[1] // pc_w)
                        depth = depth[::sh, ::sw][:pc_h, :pc_w]
            # d2c：保持流分辨率（通常 1280x1056），与 RGB 像素一一对应
            # 彩色偶发 0fps / GDK 抛错时仍出无色点云，避免整路预览失败
            color_img = None
            for _try in range(3):
                try:
                    color_img = self.camera.get_latest_image(color_type, 400.0)
                except Exception as e:
                    print(f'⚠️ 点云取彩色失败({side}): {e}')
                    color_img = None
                if color_img is not None:
                    break
            color_bgr = self._decode_color_bgr(color_img) if color_img is not None else None
            if color_bgr is not None and HAS_OPENCV:
                dh, dw = depth.shape[:2]
                if (color_bgr.shape[1], color_bgr.shape[0]) != (dw, dh):
                    color_bgr = cv2.resize(
                        color_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
            intr = self._load_hand_pc_intrinsic(
                side, depth.shape[1], depth.shape[0],
                mode=pc_mode, allow_gdk=allow_gdk_intrinsic,
            )
            pc = self.depth_to_pointcloud(
                depth, color_bgr, intr, pixel_stride=pixel_stride)
            xyz = pc['xyz']
            rot = int(getattr(self, 'hand_pc_rotate_z_deg', 0) or 0) % 360
            if rot == 180 and xyz is not None and len(xyz):
                # 绕光轴 Z 转 180°：(x,y,z) -> (-x,-y,z)
                xyz = np.asarray(xyz, dtype=np.float32).copy()
                xyz[:, 0] *= -1.0
                xyz[:, 1] *= -1.0
            return {
                'success': True,
                'side': side,
                'depth_name': depth_name,
                'xyz': xyz,
                'rgb': pc['rgb'],
                'count': pc['count'],
                'intrinsic': intr,
                'warning': intr.get('warning'),
                'depth_size': [int(depth.shape[1]), int(depth.shape[0])],
                'pc_mode': pc_mode,
                'rotate_z_deg': rot,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def build_head_pointcloud(
        self,
        index: int = -1,
        pixel_stride: int = 1,
        allow_gdk_intrinsic: bool = True,
    ) -> Dict[str, Any]:
        """构建头部 RGBD 点云（彩色内参 + D2C 对齐深度）。"""
        if self.camera is None:
            return {'success': False, 'error': '相机未初始化'}
        if index >= 0 and self._head_rgbd_pair_for_index(index) is None:
            return {'success': False, 'error': '请选择头部彩色或深度相机'}

        try:
            depth_img = self.camera.get_latest_image(
                agibot_gdk.CameraType.kHeadDepth, 200.0)
            if depth_img is None:
                return {'success': False, 'error': '无法获取头部深度'}
            if getattr(depth_img, 'color_format', None) != agibot_gdk.ColorFormat.RS2_FORMAT_Z16:
                return {'success': False, 'error': '深度格式不是 Z16'}

            depth = self._reshape_depth_u16(
                np.frombuffer(depth_img.data, dtype=np.uint16),
                int(depth_img.width or 0),
                int(depth_img.height or 0),
            )
            color_img = None
            for _try in range(3):
                try:
                    color_img = self.camera.get_latest_image(
                        agibot_gdk.CameraType.kHeadColor, 400.0)
                except Exception as e:
                    print(f'⚠️ 头部点云取彩色失败: {e}')
                    color_img = None
                if color_img is not None:
                    break
            color_bgr = self._decode_color_bgr(color_img) if color_img is not None else None
            if color_bgr is not None and HAS_OPENCV:
                dh, dw = depth.shape[:2]
                if (color_bgr.shape[1], color_bgr.shape[0]) != (dw, dh):
                    color_bgr = cv2.resize(
                        color_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
            try:
                intr = self._load_head_color_intrinsic(
                    depth.shape[1], depth.shape[0], allow_gdk=allow_gdk_intrinsic)
            except Exception as e:
                return {'success': False, 'error': f'头部内参: {e}'}
            pc = self.depth_to_pointcloud(
                depth, color_bgr, intr, pixel_stride=pixel_stride)
            return {
                'success': True,
                'side': 'head',
                'depth_name': '头部深度相机',
                'xyz': pc['xyz'],
                'rgb': pc['rgb'],
                'count': pc['count'],
                'intrinsic': intr,
                'warning': intr.get('warning'),
                'depth_size': [int(depth.shape[1]), int(depth.shape[0])],
                'pc_mode': 'd2c',
                'rotate_z_deg': 0,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def build_rgbd_pointcloud(
        self,
        index: int,
        pixel_stride: int = 1,
        allow_gdk_intrinsic: bool = True,
    ) -> Dict[str, Any]:
        """手部或头部 RGBD 点云统一入口。"""
        if self._hand_rgbd_pair_for_index(index) is not None:
            return self.build_hand_pointcloud(
                index=index,
                pixel_stride=pixel_stride,
                allow_gdk_intrinsic=allow_gdk_intrinsic,
            )
        if self._head_rgbd_pair_for_index(index) is not None:
            return self.build_head_pointcloud(
                index=index,
                pixel_stride=pixel_stride,
                allow_gdk_intrinsic=allow_gdk_intrinsic,
            )
        return {'success': False, 'error': '请选择头部或手部 RGBD 相机'}

    def pack_pointcloud_preview_bytes(
        self,
        index: int,
        pixel_stride: int = 1,
        max_points: int = 120000,
    ) -> Dict[str, Any]:
        """打包实时预览点云：AGPC 头 + float32 xyz + uint8 rgb。

        与落盘一致：深度/彩色先规整到 hand_pc_depth_size(640x480) 再用 Intr_TOF。
        """
        # 实时预览跳过 GDK 内参（易超时），用文件或 FOV 估计
        built = self.build_rgbd_pointcloud(
            index=index, pixel_stride=pixel_stride, allow_gdk_intrinsic=False)
        if not built.get('success'):
            return built

        xyz = np.asarray(built['xyz'], dtype=np.float32)
        rgb = np.asarray(built['rgb'], dtype=np.uint8)
        n = int(xyz.shape[0])
        if n > max_points > 0:
            step = max(1, int(np.ceil(n / float(max_points))))
            xyz = xyz[::step]
            rgb = rgb[::step]
            n = int(xyz.shape[0])

        # header: magic(4) + ver(u16) + flags(u16) + count(u32) + side(16 ascii)
        side = str(built.get('side') or '')[:15]
        side_bytes = side.encode('ascii', errors='ignore')[:15].ljust(16, b'\0')
        header = struct.pack('<4sHHI', b'AGPC', 1, 1, n) + side_bytes
        body = xyz.tobytes(order='C') + rgb.tobytes(order='C')
        depth_size = built.get('depth_size') or list(self.hand_pc_depth_size)
        return {
            'success': True,
            'payload': header + body,
            'count': n,
            'side': side,
            'warning': built.get('warning'),
            'depth_size': depth_size,
            'intrinsic': built.get('intrinsic'),
        }

    def capture_hand_pointcloud_to_disk(
        self,
        index: int,
        output_dir: Optional[str] = None,
        pixel_stride: int = 1,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """采集一手/头部 RGBD 点云：PLY(带色) + NPY(xyzrgb) + meta.json。"""
        built = self.build_rgbd_pointcloud(index=index, pixel_stride=pixel_stride)
        if not built.get('success'):
            return built

        side = built['side']
        depth_name = built['depth_name']
        xyz, rgb = built['xyz'], built['rgb']
        intr = built['intrinsic']
        try:
            ts = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            seq = 1
            with self._project_lock:
                proj = self.active_project
                if output_dir:
                    out_dir = output_dir
                elif proj and not proj.get('closed'):
                    out_dir = os.path.join(proj['dir'], 'pointclouds')
                    proj['capture_count'] = int(proj.get('capture_count', 0)) + 1
                    seq = proj['capture_count']
                else:
                    out_dir = os.path.join(self.save_dir, f'rgbd_pc_{side}_{ts}')
            os.makedirs(out_dir, exist_ok=True)

            stem = f'{side}_rgbd_{ts}'
            ply_path = os.path.join(out_dir, f'{stem}.ply')
            npy_path = os.path.join(out_dir, f'{stem}_xyzrgb.npy')
            meta_path = os.path.join(out_dir, f'{stem}_meta.json')

            self._write_ply_xyzrgb(ply_path, xyz, rgb)
            np.save(npy_path, np.concatenate([xyz, rgb.astype(np.float32)], axis=1))

            meta = {
                'side': side,
                'depth_camera': depth_name,
                'timestamp': ts,
                'point_count': built['count'],
                'depth_size': built.get('depth_size'),
                'intrinsic': intr,
                'unit': 'meter',
                'pixel_stride': pixel_stride,
                'rotate_z_deg': built.get('rotate_z_deg', 0),
                'pc_mode': built.get('pc_mode'),
                'files': [ply_path, npy_path, meta_path],
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            print(f'☁️ 点云已保存 {side}: {built["count"]} pts -> {ply_path}')
            return {
                'success': True,
                'side': side,
                'seq': seq,
                'point_count': built['count'],
                'files': [ply_path, npy_path, meta_path],
                'save_dir': out_dir,
                'intrinsic_source': intr.get('source'),
                'warning': intr.get('warning'),
                'status': self.get_project_status(),
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def start_hand_pointcloud_recording(
        self,
        index: int,
        interval_s: float = 0.1,
        pixel_stride: int = 2,
    ) -> Dict[str, Any]:
        """开始连续录制手部/头部点云。"""
        pair = self._rgbd_pair_for_index(index)
        if pair is None:
            return {'success': False, 'error': '请选择头部或手部 RGBD 相机'}
        _, _, side, _ = pair

        with self._pc_record_lock:
            if self._pc_record.get('active'):
                return {
                    'success': False,
                    'error': f'已在录制 {self._pc_record.get("side")} 点云',
                    'recording': self.get_pointcloud_record_status(),
                }

            with self._project_lock:
                proj = self.active_project
                if proj and not proj.get('closed'):
                    out_dir = os.path.join(proj['dir'], 'pointclouds', side)
                else:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    out_dir = os.path.join(self.save_dir, f'rgbd_pc_seq_{side}_{ts}')
            os.makedirs(out_dir, exist_ok=True)

            stop_event = threading.Event()
            self._pc_record.update({
                'active': True,
                'side': side,
                'index': index,
                'dir': out_dir,
                'count': 0,
                'interval_s': max(0.05, float(interval_s)),
                'pixel_stride': max(1, int(pixel_stride)),
                'stop_event': stop_event,
                'last_error': None,
            })

            def _loop():
                while not stop_event.is_set():
                    t0 = time.time()
                    res = self.capture_hand_pointcloud_to_disk(
                        index=index,
                        output_dir=out_dir,
                        pixel_stride=self._pc_record['pixel_stride'],
                    )
                    with self._pc_record_lock:
                        if res.get('success'):
                            self._pc_record['count'] += 1
                        else:
                            self._pc_record['last_error'] = res.get('error')
                    elapsed = time.time() - t0
                    stop_event.wait(max(0.0, self._pc_record['interval_s'] - elapsed))
                with self._pc_record_lock:
                    self._pc_record['active'] = False

            th = threading.Thread(target=_loop, name=f'pc-record-{side}', daemon=True)
            self._pc_record['thread'] = th
            th.start()

        return {
            'success': True,
            'message': f'开始连续录制 {side} 点云',
            'recording': self.get_pointcloud_record_status(),
        }

    def stop_hand_pointcloud_recording(self) -> Dict[str, Any]:
        with self._pc_record_lock:
            if not self._pc_record.get('active'):
                return {
                    'success': True,
                    'message': '未在录制',
                    'recording': self.get_pointcloud_record_status(),
                }
            ev = self._pc_record.get('stop_event')
            th = self._pc_record.get('thread')
        if ev:
            ev.set()
        if th and th.is_alive():
            th.join(timeout=5.0)
        with self._pc_record_lock:
            self._pc_record['active'] = False
        return {
            'success': True,
            'message': '已停止点云录制',
            'recording': self.get_pointcloud_record_status(),
        }

    def get_pointcloud_record_status(self) -> Dict[str, Any]:
        with self._pc_record_lock:
            r = self._pc_record
            return {
                'active': bool(r.get('active')),
                'side': r.get('side'),
                'index': r.get('index'),
                'dir': r.get('dir'),
                'count': int(r.get('count') or 0),
                'interval_s': r.get('interval_s'),
                'pixel_stride': r.get('pixel_stride'),
                'last_error': r.get('last_error'),
            }

    def _project_status_unlocked(self) -> Dict[str, Any]:
        """返回当前工程状态（调用方需持有 _project_lock）"""
        proj = self.active_project
        now = time.time()
        if not proj:
            return {
                'active': False,
                'message': '未开启采集工程',
                'default_duration_sec': self.default_capture_duration_sec,
            }

        remaining = max(0.0, proj['end_ts'] - now)
        if remaining <= 0 and not proj.get('closed'):
            proj['closed'] = True
            self._finalize_project_unlocked(proj, reason='duration_elapsed')

        if proj.get('closed'):
            return {
                'active': False,
                'message': '采集工程已结束',
                'project': {
                    'name': proj['name'],
                    'dir': proj['dir'],
                    'images_dir': proj['images_dir'],
                    'capture_count': proj['capture_count'],
                    'duration_sec': proj['duration_sec'],
                    'start_time': proj['start_time'],
                    'end_time': proj.get('end_time'),
                    'closed_reason': proj.get('closed_reason'),
                },
                'default_duration_sec': self.default_capture_duration_sec,
            }

        return {
            'active': True,
            'remaining_sec': round(remaining, 1),
            'message': f"工程进行中，剩余 {int(remaining)} 秒",
            'project': {
                'name': proj['name'],
                'dir': proj['dir'],
                'images_dir': proj['images_dir'],
                'capture_count': proj['capture_count'],
                'duration_sec': proj['duration_sec'],
                'start_time': proj['start_time'],
                'params_saved': proj.get('params_saved', False),
            },
            'default_duration_sec': self.default_capture_duration_sec,
        }

    def get_project_status(self) -> Dict[str, Any]:
        with self._project_lock:
            return self._project_status_unlocked()

    def _finalize_project_unlocked(self, proj: Dict[str, Any], reason: str):
        """写工程元数据并标记结束"""
        proj['closed'] = True
        proj['closed_reason'] = reason
        proj['end_time'] = datetime.now().isoformat(timespec='seconds')
        meta = {
            'name': proj['name'],
            'dir': proj['dir'],
            'images_dir': proj['images_dir'],
            'calib_dir': proj['calib_dir'],
            'duration_sec': proj['duration_sec'],
            'start_time': proj['start_time'],
            'end_time': proj['end_time'],
            'capture_count': proj['capture_count'],
            'params_saved': proj.get('params_saved', False),
            'closed_reason': reason,
        }
        try:
            with open(os.path.join(proj['dir'], 'project.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 写入 project.json 失败: {e}")
        print(f"📦 采集工程结束 ({reason}): {proj['dir']} | 采集次数={proj['capture_count']}")

    def start_capture_project(
        self,
        duration_sec: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """开启一个采集工程：按时长归组，时段内图像落在同一目录"""
        with self._project_lock:
            status = self._project_status_unlocked()
            if status.get('active'):
                return {
                    'success': False,
                    'error': '已有进行中的采集工程，请先结束或等待时长到期',
                    'status': status,
                }

            try:
                duration = float(
                    duration_sec if duration_sec is not None else self.default_capture_duration_sec
                )
            except (TypeError, ValueError):
                return {'success': False, 'error': '采集时长无效'}
            if duration <= 0:
                return {'success': False, 'error': '采集时长必须 > 0 秒'}
            if duration > 24 * 3600:
                return {'success': False, 'error': '采集时长过长（最大 24 小时）'}

            start_dt = datetime.now()
            ts = start_dt.strftime("%Y%m%d_%H%M%S")
            safe_name = self._safe_camera_filename(name) if name else "project"
            project_name = f"{safe_name}_{ts}"
            project_dir = os.path.join(self.save_dir, project_name)
            images_dir = os.path.join(project_dir, "images")
            calib_dir = os.path.join(project_dir, "calib")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(calib_dir, exist_ok=True)

            start_ts = time.time()
            self.active_project = {
                'name': project_name,
                'dir': project_dir,
                'images_dir': images_dir,
                'calib_dir': calib_dir,
                'start_ts': start_ts,
                'end_ts': start_ts + duration,
                'duration_sec': duration,
                'start_time': start_dt.isoformat(timespec='seconds'),
                'capture_count': 0,
                'params_saved': False,
                'closed': False,
            }

            # 工程开始时落盘一次内外参（之后采集不再重复建目录）
            param_result = self.save_head_stereo_camera_params(calib_dir)
            self.active_project['params_saved'] = bool(param_result.get('success'))
            self.active_project['param_result'] = param_result

            with open(os.path.join(project_dir, 'project.json'), 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'name': project_name,
                        'dir': project_dir,
                        'images_dir': images_dir,
                        'calib_dir': calib_dir,
                        'duration_sec': duration,
                        'start_time': self.active_project['start_time'],
                        'planned_end_ts': self.active_project['end_ts'],
                        'capture_count': 0,
                        'params_saved': self.active_project['params_saved'],
                        'param_result': {
                            'success': param_result.get('success'),
                            'error': param_result.get('error'),
                            'files': param_result.get('files', []),
                        },
                        'status': 'running',
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(
                f"📦 开启采集工程: {project_dir} | 时长={duration}s | "
                f"内外参={'OK' if param_result.get('success') else param_result.get('error')}"
            )
            return {
                'success': True,
                'status': self._project_status_unlocked(),
                'param_result': param_result,
            }

    def stop_capture_project(self) -> Dict[str, Any]:
        """手动结束当前采集工程"""
        with self._project_lock:
            status = self._project_status_unlocked()
            if not self.active_project:
                return {'success': False, 'error': '当前没有采集工程', 'status': status}
            if self.active_project.get('closed'):
                return {'success': True, 'message': '工程已结束', 'status': status}
            self._finalize_project_unlocked(self.active_project, reason='manual_stop')
            return {
                'success': True,
                'message': '已结束采集工程',
                'status': self._project_status_unlocked(),
            }

    def capture_stereo_pair_to_disk(self, index: Optional[int] = None) -> Dict[str, Any]:
        """在当前采集工程内落盘一组图像（头部双目 / 手部单目彩色 / 单路）。"""
        if self.camera is None:
            return {'success': False, 'error': '相机未初始化'}

        # 未传 index 时保持旧行为：头部双目
        if index is None:
            group = {
                'success': True,
                'kind': 'head_stereo',
                'indices': self._stereo_pair_indices(),
                'label': '头部双目',
            }
            if len(group['indices']) < 2:
                return {'success': False, 'error': '未配置头部双目左/右相机'}
        else:
            group = self._capture_group_for_index(int(index))
            if not group.get('success'):
                return group
            if not group.get('indices'):
                return {'success': False, 'error': f"未找到可落盘相机: {group.get('label')}"}

        indices = group['indices']
        label = group.get('label') or '相机'

        with self._project_lock:
            status = self._project_status_unlocked()
            if not status.get('active'):
                return {
                    'success': False,
                    'error': '请先设置采集时长并「开始工程」，再点采集落盘',
                    'status': status,
                }
            proj = self.active_project
            proj['capture_count'] += 1
            seq = proj['capture_count']
            images_dir = proj['images_dir']
            project_dir = proj['dir']

        # 锁外取图，避免阻塞其他请求
        frame_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        seq_tag = f"{seq:04d}"
        results = []
        all_files = []
        errors = []

        for cam_index in indices:
            camera_type, camera_name = self.camera_types[cam_index]
            custom_ts = f"{seq_tag}_{frame_ts}"
            result = self.capture_camera_to_disk(
                cam_index, timestamp=custom_ts, output_dir=images_dir
            )
            results.append(result)
            if result.get('success'):
                all_files.extend(result.get('files', []))
            else:
                errors.append(f"{camera_name}: {result.get('error', '未知错误')}")

        with self._project_lock:
            status = self._project_status_unlocked()
            try:
                meta_path = os.path.join(project_dir, 'project.json')
                meta = {}
                if os.path.isfile(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                meta['capture_count'] = self.active_project['capture_count'] if self.active_project else seq
                meta['last_capture_time'] = datetime.now().isoformat(timespec='seconds')
                meta['last_capture_kind'] = group.get('kind')
                meta['last_capture_label'] = label
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"⚠️ 更新 project.json 失败: {e}")

        if not all_files:
            return {
                'success': False,
                'error': f'{label}落盘失败: ' + '; '.join(errors),
                'results': results,
                'status': status,
                'save_dir': images_dir,
                'kind': group.get('kind'),
            }

        ok_names = [r['camera_name'] for r in results if r.get('success')]
        return {
            'success': True,
            'camera_name': ' + '.join(ok_names) if ok_names else label,
            'indices': indices,
            'files': all_files,
            'timestamp': frame_ts,
            'seq': seq,
            'save_dir': images_dir,
            'project_dir': project_dir,
            'partial_errors': errors,
            'results': results,
            'status': status,
            'kind': group.get('kind'),
            'label': label,
        }
    
    def update_image(self):
        """更新当前图像"""
        if self.camera is None:
            return
        
        try:
            camera_type, camera_name = self.camera_types[self.current_camera_index]
            image = self.camera.get_latest_image(camera_type, 1000.0)
            
            if image is not None:
                decoded_image = self.decode_image_data(image)
                if decoded_image is not None:
                    self.current_image = decoded_image
                else:
                    print(f"❌ 无法解码 {camera_name} 图像")
            else:
                print(f"⏳ 等待 {camera_name} 图像...")
        except Exception as e:
            print(f"更新图像失败: {e}")
    
    def setup_routes(self):
        """设置Flask路由"""
        
        # HTML模板
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Genie02图像实时画面</title>
    <meta charset="utf-8">
    <script type="importmap">
    {
      "imports": {
        "three": "https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/"
      }
    }
    </script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            padding: 20px; 
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header { 
            text-align: center; 
            margin-bottom: 20px; 
            color: #333;
        }
        .toolbar {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
            padding: 14px;
            background: #f7f7f7;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        .project-panel {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 16px;
            padding: 14px;
            background: #eef6ff;
            border: 1px solid #c5daf5;
            border-radius: 8px;
        }
        .project-panel label {
            font-size: 14px;
            color: #333;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .project-panel input[type="number"],
        .project-panel input[type="text"] {
            padding: 6px 8px;
            border: 1px solid #bbb;
            border-radius: 6px;
            width: 120px;
        }
        .project-status {
            flex: 1 1 100%;
            font-size: 14px;
            color: #1f4e79;
            margin-top: 4px;
        }
        .camera-select-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px 16px;
            flex: 1;
        }
        .camera-select-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            color: #333;
            cursor: pointer;
            user-select: none;
        }
        .toolbar-actions {
            display: flex;
            gap: 8px;
            flex-shrink: 0;
        }
        .btn {
            border: 1px solid #bbb;
            background: #fff;
            border-radius: 6px;
            padding: 8px 14px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn:hover { background: #eee; }
        .btn-primary {
            background: #2f6fed;
            border-color: #2f6fed;
            color: #fff;
        }
        .btn-primary:hover { background: #1f5bd6; }
        .btn-danger {
            background: #b42318;
            border-color: #b42318;
            color: #fff;
        }
        .btn-danger:hover { background: #912018; }
        .btn-capture {
            background: #1a7f37;
            border-color: #1a7f37;
            color: #fff;
            margin-top: 10px;
            width: 100%;
        }
        .btn-capture:hover { background: #146c2e; }
        .btn-capture:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .btn-pc {
            background: #8250df;
            border-color: #8250df;
            color: #fff;
            margin-top: 0;
            flex: 1 1 140px;
            min-width: 140px;
        }
        .btn-pc:hover { background: #6b3fc7; }
        .btn-pc-rec {
            background: #bf8700;
            border-color: #bf8700;
            color: #fff;
            margin-top: 0;
            flex: 1 1 140px;
            min-width: 140px;
        }
        .btn-pc-rec:hover { background: #9a6b00; }
        .btn-pc-rec.active {
            background: #b42318;
            border-color: #b42318;
        }
        .btn-pc-live {
            background: #0969da;
            border-color: #0969da;
            color: #fff;
            margin-top: 0;
            flex: 1 1 140px;
            min-width: 140px;
        }
        .btn-pc-live:hover { background: #0550ae; }
        .btn-pc-live.active {
            background: #1a7f37;
            border-color: #1a7f37;
        }
        .camera-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .camera-actions .btn-capture {
            margin-top: 0;
            flex: 1 1 100%;
        }
        .pc-viewer-panel {
            display: none;
            margin: 16px 0 24px;
            border: 1px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
            background: #111;
        }
        .pc-viewer-panel.active { display: block; }
        .pc-viewer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 10px 14px;
            background: #1f2328;
            color: #fff;
            font-size: 14px;
        }
        .pc-viewer-host {
            width: 100%;
            height: 460px;
            touch-action: none;
        }
        .pc-viewer-meta {
            padding: 8px 14px;
            background: #161b22;
            color: #8b949e;
            font-size: 13px;
        }
        .btn-d2c-hand {
            background: #8250df;
            border-color: #8250df;
            color: #fff;
            margin-top: 0;
            flex: 1 1 140px;
            min-width: 140px;
        }
        .btn-d2c-hand:hover { background: #6639ba; }
        .btn-d2c-hand.active {
            background: #1a7f37;
            border-color: #1a7f37;
        }
        .btn-d2c-head {
            background: #8250df;
            border-color: #8250df;
            color: #fff;
            margin-top: 0;
            flex: 1 1 140px;
            min-width: 140px;
        }
        .btn-d2c-head:hover { background: #6639ba; }
        .btn-d2c-head.active {
            background: #1a7f37;
            border-color: #1a7f37;
        }
        .hand-d2c-panel {
            display: none;
            margin: 16px 0 24px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background: #fafafa;
            overflow: hidden;
        }
        .hand-d2c-panel.active { display: block; }
        .hand-d2c-body {
            position: relative;
            padding: 12px 14px 16px;
            text-align: center;
            background: #111;
            cursor: crosshair;
            user-select: none;
        }
        .hand-d2c-body img {
            max-width: 100%;
            max-height: 560px;
            border: 1px solid #333;
            border-radius: 6px;
            background: #000;
            display: inline-block;
            vertical-align: top;
        }
        .hand-d2c-crosshair {
            position: absolute;
            width: 14px;
            height: 14px;
            margin: -7px 0 0 -7px;
            border: 2px solid #ffdd57;
            border-radius: 50%;
            box-shadow: 0 0 0 1px #000;
            pointer-events: none;
            display: none;
            z-index: 2;
        }
        .hand-d2c-depth-label {
            position: absolute;
            transform: translate(12px, -120%);
            padding: 4px 8px;
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.78);
            color: #ffdd57;
            font: 600 14px/1.3 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            white-space: nowrap;
            pointer-events: none;
            display: none;
            z-index: 3;
            border: 1px solid rgba(255, 221, 87, 0.55);
            text-shadow: 0 1px 2px #000;
        }
        .hand-d2c-depth-label.invalid {
            color: #ff7b72;
            border-color: rgba(255, 123, 114, 0.55);
        }
        .d2c-panel {
            display: none;
            margin: 16px 0 24px;
            border: 1px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
            background: #fafafa;
        }
        .d2c-panel.active { display: block; }
        .d2c-header {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            padding: 12px 14px;
            background: #1f2328;
            color: #fff;
            font-size: 15px;
            font-weight: 600;
        }
        .d2c-controls {
            display: flex;
            flex-wrap: wrap;
            gap: 12px 18px;
            padding: 12px 14px;
            align-items: center;
            font-size: 13px;
            color: #333;
            border-bottom: 1px solid #eee;
        }
        .d2c-controls label {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .d2c-controls input[type="range"] { width: 120px; }
        .d2c-controls input[type="number"] { width: 72px; }
        .d2c-controls select { padding: 4px 8px; }
        .d2c-body {
            position: relative;
            padding: 12px 14px 16px;
            text-align: center;
            background: #111;
            cursor: crosshair;
            user-select: none;
        }
        .d2c-body img {
            max-width: 100%;
            max-height: 520px;
            border: 1px solid #333;
            border-radius: 6px;
            background: #000;
            display: inline-block;
            vertical-align: top;
        }
        .d2c-meta {
            padding: 8px 14px 12px;
            font-size: 13px;
            color: #666;
            background: #f5f5f5;
        }
        .cameras-grid { 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .camera-card { 
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            background-color: #fafafa;
            text-align: center;
        }
        .camera-title { 
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .camera-info { 
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        .camera-image { 
            max-width: 100%; 
            max-height: 360px; 
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: #000;
        }
        .capture-msg {
            margin-top: 8px;
            font-size: 13px;
            color: #1a7f37;
            min-height: 18px;
            word-break: break-all;
        }
        .status { 
            text-align: center; 
            margin-top: 20px; 
            padding: 15px;
            background-color: #d4edda;
            border-radius: 5px;
            color: #155724;
            font-size: 16px;
        }
        .refresh-info {
            text-align: center;
            margin-bottom: 16px;
            color: #666;
            font-style: italic;
        }
        .empty-hint {
            text-align: center;
            padding: 50px;
            color: #666;
            border: 1px dashed #ccc;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>机器人相机查看器</h1>
        </div>

        <div class="project-panel">
            <label>工程名
                <input type="text" id="project-name" placeholder="project" value="project">
            </label>
            <label>采集时长(秒)
                <input type="number" id="capture-duration" min="1" step="1" value="300">
            </label>
            <button class="btn btn-primary" id="btn-start-project" onclick="startProject()">开始工程</button>
            <button class="btn btn-danger" id="btn-stop-project" onclick="stopProject()">结束工程</button>
            <div class="project-status" id="project-status">工程状态加载中...</div>
        </div>
        
        <div class="toolbar">
            <div class="camera-select-list" id="camera-select-list">加载相机列表中...</div>
            <div class="toolbar-actions">
                <button class="btn" onclick="selectAllCameras()">全选</button>
                <button class="btn" onclick="clearCameraSelection()">清空</button>
            </div>
        </div>

        <div class="refresh-info">
            <p><!--HAND_CAPTURE_HINT--></p>
        </div>

        <div class="d2c-panel" id="d2c-panel">
            <div class="d2c-header">
                <span>头部 RGBD · 深度叠彩图 / 点选测深</span>
                <button class="btn" type="button" id="d2c-toggle-btn" onclick="toggleHeadD2C()">暂停</button>
            </div>
            <div class="d2c-controls">
                <label>模式
                    <select id="d2c-mode" onchange="refreshHeadD2C(true)">
                        <option value="overlay">深度叠彩图</option>
                        <option value="color">纯彩色(可点选)</option>
                        <option value="side">三栏对比</option>
                    </select>
                </label>
                <label>透明度 <span id="d2c-alpha-val">0.45</span>
                    <input type="range" id="d2c-alpha" min="0" max="1" step="0.05" value="0.45"
                           oninput="document.getElementById('d2c-alpha-val').textContent=this.value"
                           onchange="refreshHeadD2C(true)">
                </label>
                <label>近距(mm)
                    <input type="number" id="d2c-zmin" value="200" min="0" step="50"
                           onchange="refreshHeadD2C(true)">
                </label>
                <label>远距(mm)
                    <input type="number" id="d2c-zmax" value="3000" min="100" step="100"
                           onchange="refreshHeadD2C(true)">
                </label>
                <label>
                    <input type="checkbox" id="d2c-edges" onchange="refreshHeadD2C(true)"> 深度边缘
                </label>
                <label>棋盘列
                    <input type="number" id="d2c-board-cols" value="9" min="2" max="30" step="1">
                </label>
                <label>棋盘行
                    <input type="number" id="d2c-board-rows" value="6" min="2" max="30" step="1">
                </label>
                <label>方格(mm)
                    <input type="number" id="d2c-square-mm" value="25" min="1" step="0.5">
                </label>
                <button class="btn" type="button" id="d2c-plane-btn" onclick="runHeadPlaneMetric()">测平面</button>
            </div>
            <div class="d2c-body" id="d2c-body">
                <img id="d2c-image" alt="头部 D2C" draggable="false">
                <div class="hand-d2c-crosshair" id="d2c-crosshair"></div>
                <div class="hand-d2c-depth-label" id="d2c-depth-label"></div>
            </div>
            <div class="d2c-meta" id="d2c-meta">点击图像任意位置读取该点深度（mm）</div>
            <div class="d2c-meta" id="d2c-probe">尚未点选</div>
            <div class="d2c-meta" id="d2c-plane-meta">平面指标：尚未测量（将棋盘放入视野后点「测平面」）</div>
        </div>
        
        <div class="cameras-grid" id="cameras-grid">
            <div class="empty-hint">请先勾选要显示的相机</div>
        </div>

        <div class="pc-viewer-panel" id="pc-viewer-panel">
            <div class="pc-viewer-header">
                <span id="pc-viewer-title">RGBD 点云实时预览</span>
                <button class="btn" onclick="stopLivePointCloud()">关闭预览</button>
            </div>
            <div class="pc-viewer-host" id="pc-viewer-host"></div>
            <div class="pc-viewer-meta" id="pc-viewer-meta">拖拽旋转 / 滚轮缩放</div>
        </div>

        <div class="hand-d2c-panel" id="hand-d2c-panel">
            <div class="d2c-header">
                <span id="hand-d2c-title">手部 · 深度映射到彩色</span>
                <button class="btn" type="button" onclick="stopHandD2C()">关闭</button>
            </div>
            <div class="d2c-controls">
                <label>模式
                    <select id="hand-d2c-mode" onchange="refreshHandD2C(true)">
                        <option value="overlay">深度叠彩图</option>
                        <option value="color">纯彩色(可点选)</option>
                        <option value="side">三栏对比</option>
                    </select>
                </label>
                <label>透明度 <span id="hand-d2c-alpha-val">0.45</span>
                    <input type="range" id="hand-d2c-alpha" min="0" max="1" step="0.05" value="0.45"
                           oninput="document.getElementById('hand-d2c-alpha-val').textContent=this.value"
                           onchange="refreshHandD2C(true)">
                </label>
                <label>近距(mm)
                    <input type="number" id="hand-d2c-zmin" value="200" min="0" step="50"
                           onchange="refreshHandD2C(true)">
                </label>
                <label>远距(mm)
                    <input type="number" id="hand-d2c-zmax" value="3000" min="100" step="100"
                           onchange="refreshHandD2C(true)">
                </label>
                <label>
                    <input type="checkbox" id="hand-d2c-edges" onchange="refreshHandD2C(true)"> 深度边缘
                </label>
            </div>
            <div class="hand-d2c-body" id="hand-d2c-body">
                <img id="hand-d2c-image" alt="手部 D2C" draggable="false">
                <div class="hand-d2c-crosshair" id="hand-d2c-crosshair"></div>
                <div class="hand-d2c-depth-label" id="hand-d2c-depth-label"></div>
            </div>
            <div class="d2c-meta" id="hand-d2c-meta">点击图像任意位置读取该点深度（mm）</div>
            <div class="d2c-meta" id="hand-d2c-probe">尚未点选</div>
        </div>
        
        <div class="status">
            <p>状态: <span id="status">连接中...</span></p>
        </div>
    </div>

    <script>
        /*HAND_RGBD_MODE*/
        let availableCameras = [];
        let selectedIndices = new Set();
        let refreshing = false;
        let projectActive = false;
        let d2cActive = false;  // 头部 D2C 功能关闭
        let d2cBusy = false;
        let d2cTimer = null;
        let handD2c = {
            active: false,
            index: null,
            timer: null,
            busy: false,
            lastMeta: null,
        };

        function handD2cQuery() {
            const alpha = document.getElementById('hand-d2c-alpha').value;
            const zmin = document.getElementById('hand-d2c-zmin').value;
            const zmax = document.getElementById('hand-d2c-zmax').value;
            const mode = document.getElementById('hand-d2c-mode').value;
            const edges = document.getElementById('hand-d2c-edges').checked ? 1 : 0;
            return `alpha=${alpha}&z_min=${zmin}&z_max=${zmax}&mode=${mode}&edges=${edges}`;
        }

        function refreshHandD2C(forceMeta) {
            if (!handD2c.active || handD2c.busy || handD2c.index === null) return;
            handD2c.busy = true;
            const ts = Date.now();
            const img = document.getElementById('hand-d2c-image');
            const metaEl = document.getElementById('hand-d2c-meta');
            const url = `/hand_d2c/${handD2c.index}?${handD2cQuery()}&t=${ts}`;
            const probe = new Image();
            probe.onload = () => {
                if (img) img.src = url;
                handD2c.busy = false;
            };
            probe.onerror = () => {
                if (metaEl) metaEl.textContent = '手部 D2C 取帧失败（检查彩色/深度是否在线）';
                handD2c.busy = false;
            };
            probe.src = url;
            if (forceMeta) {
                fetch(`/hand_d2c/${handD2c.index}/meta?t=${ts}`)
                    .then(r => r.json())
                    .then(m => {
                        handD2c.lastMeta = m;
                        if (!metaEl || !m || !m.success) return;
                        metaEl.textContent =
                            `${m.side || ''} | 彩色 ${m.color_size?.[0]}x${m.color_size?.[1]} | ` +
                            `深度 ${m.depth_native_size?.[0]}x${m.depth_native_size?.[1]} | ` +
                            `对齐=${m.size_matched} | 有效深度 ${(100 * (m.valid_depth_ratio || 0)).toFixed(1)}% | ` +
                            `点击图像测深度`;
                    })
                    .catch(() => {});
            }
        }

        function setHandD2cButtons(activeIndex) {
            document.querySelectorAll('[id^="hand-d2c-btn-"]').forEach(btn => {
                const on = activeIndex !== null && btn.id === `hand-d2c-btn-${activeIndex}`;
                btn.classList.toggle('active', on);
                btn.textContent = on ? '停止深度叠彩图' : '深度叠彩图 / 点选测深';
            });
        }

        function startHandD2C(index) {
            handD2c.active = true;
            handD2c.index = index;
            const panel = document.getElementById('hand-d2c-panel');
            const title = document.getElementById('hand-d2c-title');
            const cam = availableCameras.find(c => c.index === index);
            if (title) title.textContent = `手部 · 深度映射到彩色 (${cam ? cam.name : '#' + index})`;
            if (panel) panel.classList.add('active');
            setHandD2cButtons(index);
            const probeEl = document.getElementById('hand-d2c-probe');
            if (probeEl) probeEl.textContent = '尚未点选 — 在图像上单击读取深度';
            const cross = document.getElementById('hand-d2c-crosshair');
            if (cross) cross.style.display = 'none';
            const depthLabel = document.getElementById('hand-d2c-depth-label');
            if (depthLabel) {
                depthLabel.style.display = 'none';
                depthLabel.textContent = '';
            }
            refreshHandD2C(true);
            if (handD2c.timer) clearInterval(handD2c.timer);
            handD2c.timer = setInterval(() => refreshHandD2C(false), 250);
            panel?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        function stopHandD2C() {
            handD2c.active = false;
            handD2c.index = null;
            if (handD2c.timer) {
                clearInterval(handD2c.timer);
                handD2c.timer = null;
            }
            setHandD2cButtons(null);
            const panel = document.getElementById('hand-d2c-panel');
            if (panel) panel.classList.remove('active');
        }

        function toggleHandD2C(index) {
            if (handD2c.active && handD2c.index === index) {
                stopHandD2C();
            } else {
                startHandD2C(index);
            }
        }

        function placeHandD2cMarkers(clientX, clientY) {
            const body = document.getElementById('hand-d2c-body');
            const cross = document.getElementById('hand-d2c-crosshair');
            const depthLabel = document.getElementById('hand-d2c-depth-label');
            if (!body) return;
            const br = body.getBoundingClientRect();
            const left = clientX - br.left;
            const top = clientY - br.top;
            if (cross) {
                cross.style.left = left + 'px';
                cross.style.top = top + 'px';
                cross.style.display = 'block';
            }
            if (depthLabel) {
                // 靠右边缘时改向左侧显示，避免被裁切
                const flip = left > br.width * 0.72;
                depthLabel.style.left = left + 'px';
                depthLabel.style.top = top + 'px';
                depthLabel.style.transform = flip
                    ? 'translate(calc(-100% - 12px), -120%)'
                    : 'translate(12px, -120%)';
                depthLabel.style.display = 'block';
            }
        }

        function onHandD2cClick(ev) {
            if (!handD2c.active || handD2c.index === null) return;
            const img = document.getElementById('hand-d2c-image');
            if (!img || !img.naturalWidth) return;
            const mode = document.getElementById('hand-d2c-mode')?.value || 'overlay';
            const rect = img.getBoundingClientRect();
            const x = ev.clientX - rect.left;
            const y = ev.clientY - rect.top;
            if (x < 0 || y < 0 || x > rect.width || y > rect.height) return;

            // side 模式：三栏，只取左栏彩色区域做点选
            let nx = x / rect.width;
            let ny = y / rect.height;
            if (mode === 'side') {
                const gapFrac = 8 / Math.max(1, img.naturalWidth);
                const panel = (1 - 2 * gapFrac) / 3;
                if (nx > panel) {
                    document.getElementById('hand-d2c-probe').textContent =
                        '三栏模式下请点击左侧「Color」区域测深';
                    return;
                }
                nx = nx / panel;
            }

            placeHandD2cMarkers(ev.clientX, ev.clientY);
            const depthLabel = document.getElementById('hand-d2c-depth-label');
            if (depthLabel) {
                depthLabel.classList.remove('invalid');
                depthLabel.textContent = '…';
            }

            const probeEl = document.getElementById('hand-d2c-probe');
            if (probeEl) probeEl.textContent = '读取深度中…';
            fetch(`/hand_d2c/${handD2c.index}/probe?nx=${nx.toFixed(6)}&ny=${ny.toFixed(6)}&t=${Date.now()}`)
                .then(r => r.json())
                .then(d => {
                    if (!probeEl) return;
                    if (!d.success) {
                        probeEl.textContent = '测深失败: ' + (d.error || 'unknown');
                        if (depthLabel) {
                            depthLabel.classList.add('invalid');
                            depthLabel.textContent = 'N/A';
                        }
                        return;
                    }
                    const z = d.z_mm_median5 || d.z_mm || 0;
                    const tip = d.valid
                        ? `深度 ${z} mm（像素 ${d.z_mm} mm）`
                        : '该点无有效深度（0）';
                    probeEl.textContent =
                        `${d.side || ''} | 像素 (${d.u}, ${d.v}) / ${d.color_size?.[0]}x${d.color_size?.[1]} | ` +
                        `${tip} | RGB(${(d.rgb || []).join(',')})`;
                    if (depthLabel) {
                        if (d.valid) {
                            depthLabel.classList.remove('invalid');
                            depthLabel.textContent = `${z} mm`;
                        } else {
                            depthLabel.classList.add('invalid');
                            depthLabel.textContent = '无深度';
                        }
                    }
                })
                .catch(e => {
                    if (probeEl) probeEl.textContent = '测深失败: ' + e;
                    if (depthLabel) {
                        depthLabel.classList.add('invalid');
                        depthLabel.textContent = 'ERR';
                    }
                });
        }

        function d2cQuery() {
            const alpha = document.getElementById('d2c-alpha').value;
            const zmin = document.getElementById('d2c-zmin').value;
            const zmax = document.getElementById('d2c-zmax').value;
            const mode = document.getElementById('d2c-mode').value;
            const edges = document.getElementById('d2c-edges').checked ? 1 : 0;
            return `alpha=${alpha}&z_min=${zmin}&z_max=${zmax}&mode=${mode}&edges=${edges}`;
        }

        function refreshHeadD2C(forceMeta) {
            if (!d2cActive || d2cBusy) return;
            d2cBusy = true;
            const ts = Date.now();
            const img = document.getElementById('d2c-image');
            const metaEl = document.getElementById('d2c-meta');
            const url = `/d2c/head?${d2cQuery()}&t=${ts}`;
            const probe = new Image();
            probe.onload = () => {
                img.src = url;
                d2cBusy = false;
            };
            probe.onerror = () => {
                if (metaEl) metaEl.textContent = 'D2C 取帧失败（检查头部彩色/深度是否在线）';
                d2cBusy = false;
            };
            probe.src = url;
            if (forceMeta || ts % 5 < 1) {
                fetch(`/d2c/head/meta?t=${ts}`)
                    .then(r => r.json())
                    .then(m => {
                        if (!metaEl || !m.success) {
                            if (metaEl && m && m.error) metaEl.textContent = 'meta: ' + m.error;
                            return;
                        }
                        const dt = (m.delta_ts_ms == null) ? '-' : (m.delta_ts_ms + ' ms');
                        const vr = (m.valid_depth_ratio == null)
                            ? '-' : ((m.valid_depth_ratio * 100).toFixed(1) + '%');
                        metaEl.textContent =
                            `彩色 ${m.color_size?.[0]}x${m.color_size?.[1]} | ` +
                            `深度 ${m.depth_size?.[0]}x${m.depth_size?.[1]} | ` +
                            `Δt ${dt} | 有效深度 ${vr} | 点击图像测深度`;
                    })
                    .catch(() => {});
            }
        }

        function placeHeadD2cMarkers(clientX, clientY) {
            const body = document.getElementById('d2c-body');
            const cross = document.getElementById('d2c-crosshair');
            const depthLabel = document.getElementById('d2c-depth-label');
            if (!body) return;
            const br = body.getBoundingClientRect();
            const left = clientX - br.left;
            const top = clientY - br.top;
            if (cross) {
                cross.style.left = left + 'px';
                cross.style.top = top + 'px';
                cross.style.display = 'block';
            }
            if (depthLabel) {
                const flip = left > br.width * 0.72;
                depthLabel.style.left = left + 'px';
                depthLabel.style.top = top + 'px';
                depthLabel.style.transform = flip
                    ? 'translate(calc(-100% - 12px), -120%)'
                    : 'translate(12px, -120%)';
                depthLabel.style.display = 'block';
            }
        }

        function onHeadD2cClick(ev) {
            if (!d2cActive) return;
            const img = document.getElementById('d2c-image');
            if (!img || !img.naturalWidth) return;
            const mode = document.getElementById('d2c-mode')?.value || 'overlay';
            const rect = img.getBoundingClientRect();
            const x = ev.clientX - rect.left;
            const y = ev.clientY - rect.top;
            if (x < 0 || y < 0 || x > rect.width || y > rect.height) return;

            let nx = x / rect.width;
            let ny = y / rect.height;
            if (mode === 'side') {
                const gapFrac = 8 / Math.max(1, img.naturalWidth);
                const panel = (1 - 2 * gapFrac) / 3;
                if (nx > panel) {
                    document.getElementById('d2c-probe').textContent =
                        '三栏模式下请点击左侧「Color」区域测深';
                    return;
                }
                nx = nx / panel;
            }

            placeHeadD2cMarkers(ev.clientX, ev.clientY);
            const depthLabel = document.getElementById('d2c-depth-label');
            if (depthLabel) {
                depthLabel.classList.remove('invalid');
                depthLabel.textContent = '…';
            }
            const probeEl = document.getElementById('d2c-probe');
            if (probeEl) probeEl.textContent = '读取深度中…';
            fetch(`/d2c/head/probe?nx=${nx.toFixed(6)}&ny=${ny.toFixed(6)}&t=${Date.now()}`)
                .then(r => r.json())
                .then(d => {
                    if (!probeEl) return;
                    if (!d.success) {
                        probeEl.textContent = '测深失败: ' + (d.error || 'unknown');
                        if (depthLabel) {
                            depthLabel.classList.add('invalid');
                            depthLabel.textContent = 'N/A';
                        }
                        return;
                    }
                    const z = d.z_mm_median5 || d.z_mm || 0;
                    const tip = d.valid
                        ? `深度 ${z} mm（像素 ${d.z_mm} mm）`
                        : '该点无有效深度（0）';
                    probeEl.textContent =
                        `head | 像素 (${d.u}, ${d.v}) / ${d.color_size?.[0]}x${d.color_size?.[1]} | ` +
                        `${tip} | RGB(${(d.rgb || []).join(',')})`;
                    if (depthLabel) {
                        if (d.valid) {
                            depthLabel.classList.remove('invalid');
                            depthLabel.textContent = `${z} mm`;
                        } else {
                            depthLabel.classList.add('invalid');
                            depthLabel.textContent = '无深度';
                        }
                    }
                })
                .catch(e => {
                    if (probeEl) probeEl.textContent = '测深失败: ' + e;
                    if (depthLabel) {
                        depthLabel.classList.add('invalid');
                        depthLabel.textContent = 'ERR';
                    }
                });
        }

        function startHeadD2C() {
            d2cActive = true;
            const btn = document.getElementById('d2c-toggle-btn');
            if (btn) btn.textContent = '暂停';
            const probeEl = document.getElementById('d2c-probe');
            if (probeEl) probeEl.textContent = '尚未点选 — 在图像上单击读取深度';
            const cross = document.getElementById('d2c-crosshair');
            if (cross) cross.style.display = 'none';
            const depthLabel = document.getElementById('d2c-depth-label');
            if (depthLabel) {
                depthLabel.style.display = 'none';
                depthLabel.textContent = '';
            }
            refreshHeadD2C(true);
            if (d2cTimer) clearInterval(d2cTimer);
            d2cTimer = setInterval(() => refreshHeadD2C(false), 250);
        }

        window.showHeadD2C = function() {
            const panel = document.getElementById('d2c-panel');
            if (panel) {
                panel.classList.add('active');
                panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
            document.querySelectorAll('[id^="head-d2c-btn-"]').forEach(b => {
                b.classList.add('active');
                b.textContent = '停止深度叠彩图';
            });
            startHeadD2C();
        };

        function hideHeadD2C() {
            d2cActive = false;
            if (d2cTimer) {
                clearInterval(d2cTimer);
                d2cTimer = null;
            }
            const panel = document.getElementById('d2c-panel');
            if (panel) panel.classList.remove('active');
            document.querySelectorAll('[id^="head-d2c-btn-"]').forEach(b => {
                b.classList.remove('active');
                b.textContent = '深度叠彩图 / 点选测深';
            });
        }

        function toggleHeadD2C() {
            if (!document.getElementById('d2c-panel')?.classList.contains('active')) {
                window.showHeadD2C();
                return;
            }
            d2cActive = !d2cActive;
            const btn = document.getElementById('d2c-toggle-btn');
            if (btn) btn.textContent = d2cActive ? '暂停' : '继续';
            if (d2cActive) {
                refreshHeadD2C(true);
                if (!d2cTimer) d2cTimer = setInterval(() => refreshHeadD2C(false), 250);
            } else if (d2cTimer) {
                clearInterval(d2cTimer);
                d2cTimer = null;
            }
        }

        window.toggleHeadD2CPanel = function() {
            const panel = document.getElementById('d2c-panel');
            if (panel && panel.classList.contains('active') && d2cActive) {
                hideHeadD2C();
            } else {
                window.showHeadD2C();
            }
        };

        function runHeadPlaneMetric() {
            const metaEl = document.getElementById('d2c-plane-meta');
            const btn = document.getElementById('d2c-plane-btn');
            const cols = document.getElementById('d2c-board-cols').value;
            const rows = document.getElementById('d2c-board-rows').value;
            const square = document.getElementById('d2c-square-mm').value;
            const zmin = document.getElementById('d2c-zmin').value;
            const zmax = document.getElementById('d2c-zmax').value;
            const q = new URLSearchParams({
                cols, rows, square_mm: square,
                z_min: zmin, z_max: zmax, vis: '1'
            });
            if (btn) { btn.disabled = true; btn.textContent = '测量中...'; }
            if (metaEl) metaEl.textContent = '平面指标：检测棋盘并拟合中...';
            fetch(`/d2c/head/plane_metric?${q.toString()}`)
                .then(r => r.json())
                .then(m => {
                    if (!m.success) {
                        if (metaEl) metaEl.textContent = '平面指标失败: ' + (m.error || 'unknown');
                        return;
                    }
                    const rmse = m.residual_mm ? m.residual_mm.rmse : '-';
                    const pass = m.pass
                        ? `angle=${m.pass.angle} residual=${m.pass.residual_rmse} reproj=${m.pass.reproj}`
                        : '-';
                    if (metaEl) {
                        metaEl.textContent =
                            `平面 OK | 夹角 ${m.angle_deg}° | 残差RMSE ${rmse} mm | ` +
                            `重投影 ${m.reproj_err_px} px | 距 ${m.board_distance_mm} mm | ` +
                            `点 ${m.num_points} | pass{${pass}}`;
                    }
                    if (m.vis_jpeg_b64) {
                        const img = document.getElementById('d2c-image');
                        if (img) img.src = 'data:image/jpeg;base64,' + m.vis_jpeg_b64;
                    }
                })
                .catch(e => {
                    if (metaEl) metaEl.textContent = '平面指标请求失败: ' + e;
                })
                .finally(() => {
                    if (btn) { btn.disabled = false; btn.textContent = '测平面'; }
                });
        }

        function renderProjectStatus(data) {
            const el = document.getElementById('project-status');
            if (!data) {
                el.textContent = '无法获取工程状态';
                projectActive = false;
                return;
            }
            projectActive = !!data.active;
            if (data.active && data.project) {
                el.textContent =
                    `进行中: ${data.project.name} | 剩余 ${Math.floor(data.remaining_sec)}s | 已采集 ${data.project.capture_count} 次 | 目录 ${data.project.dir}`;
            } else if (data.project) {
                el.textContent =
                    `已结束: ${data.project.name} | 共采集 ${data.project.capture_count} 次 | ${data.project.dir}`;
            } else {
                el.textContent = data.message || '未开启采集工程';
            }
            document.querySelectorAll('.btn-capture').forEach((btn) => {
                btn.disabled = !projectActive;
                btn.title = projectActive ? '' : '请先开始工程';
            });
        }

        function refreshProjectStatus() {
            fetch('/project/status')
                .then(r => r.json())
                .then(data => renderProjectStatus(data.success === false ? null : data))
                .catch(() => renderProjectStatus(null));
        }

        function startProject() {
            const duration = Number(document.getElementById('capture-duration').value);
            const name = document.getElementById('project-name').value.trim() || 'project';
            fetch('/project/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({duration_sec: duration, name: name})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    renderProjectStatus(data.status);
                    document.getElementById('status').textContent =
                        `工程已开始: ${data.status.project.dir}`;
                    if (data.param_result && !data.param_result.success) {
                        alert('工程已开始，但内外参落盘失败: ' + data.param_result.error);
                    }
                } else {
                    alert(data.error || '开始工程失败');
                    if (data.status) renderProjectStatus(data.status);
                }
            })
            .catch(err => alert('开始工程失败: ' + err));
        }

        function stopProject() {
            fetch('/project/stop', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.status) renderProjectStatus(data.status);
                    document.getElementById('status').textContent = data.message || (data.error || '已结束');
                })
                .catch(err => alert('结束工程失败: ' + err));
        }

        function getAvailableCameras() {
            fetch('/get_available_cameras')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        availableCameras = data.cameras;
                        renderCameraSelector();
                        if (availableCameras.length > 0) {
                            selectedIndices = new Set([availableCameras[0].index]);
                            syncSelectorChecks();
                        }
                        createCameraCards();
                        document.getElementById('status').textContent =
                            `检测到 ${availableCameras.length} 个可用相机，已选 ${selectedIndices.size} 个`;
                        updateSelectedImages();
                        refreshProjectStatus();
                        maybeAutoStartLivePointCloud();
                    } else {
                        document.getElementById('status').textContent = '获取相机列表失败: ' + data.error;
                    }
                })
                .catch(error => {
                    document.getElementById('status').textContent = '连接错误: ' + error;
                });
        }

        function renderCameraSelector() {
            const list = document.getElementById('camera-select-list');
            list.innerHTML = '';
            if (availableCameras.length === 0) {
                list.textContent = '没有可用相机';
                return;
            }
            availableCameras.forEach((camera) => {
                const label = document.createElement('label');
                label.className = 'camera-select-item';
                label.innerHTML = `
                    <input type="checkbox" data-index="${camera.index}" onchange="onCameraSelectionChange()">
                    <span>${camera.name}</span>
                `;
                list.appendChild(label);
            });
            syncSelectorChecks();
        }

        function syncSelectorChecks() {
            document.querySelectorAll('#camera-select-list input[type="checkbox"]').forEach((el) => {
                const idx = Number(el.dataset.index);
                el.checked = selectedIndices.has(idx);
            });
        }

        function onCameraSelectionChange() {
            selectedIndices = new Set();
            document.querySelectorAll('#camera-select-list input[type="checkbox"]').forEach((el) => {
                if (el.checked) selectedIndices.add(Number(el.dataset.index));
            });
            createCameraCards();
            updateSelectedImages();
            document.getElementById('status').textContent =
                `已选 ${selectedIndices.size} / ${availableCameras.length} 个相机`;
        }

        function selectAllCameras() {
            selectedIndices = new Set(availableCameras.map(c => c.index));
            syncSelectorChecks();
            createCameraCards();
            updateSelectedImages();
        }

        function clearCameraSelection() {
            selectedIndices = new Set();
            syncSelectorChecks();
            createCameraCards();
        }

        function getSelectedCameraObjects() {
            return availableCameras.filter(c => selectedIndices.has(c.index));
        }

        function createCameraCards() {
            const grid = document.getElementById('cameras-grid');
            grid.innerHTML = '';
            const selected = getSelectedCameraObjects();

            if (availableCameras.length === 0) {
                grid.innerHTML = '<div class="empty-hint">没有检测到可用的相机</div>';
                return;
            }
            if (selected.length === 0) {
                grid.innerHTML = '<div class="empty-hint">请先勾选要显示的相机</div>';
                return;
            }

            selected.forEach((camera) => {
                const card = document.createElement('div');
                card.className = 'camera-card';
                card.id = `card-${camera.index}`;
                // 机型显示名可能是「左手/右手」或「左腕部/右腕部」
                const isHand = /手|腕/.test(camera.name);
                const isHandDepth = isHand && camera.name.includes('深度');
                const isHandColor = isHand && /彩色|color/i.test(camera.name);
                const isHeadRgbd = camera.name.includes('头部RGBD');
                const isHeadStereo = camera.name.includes('头部双目');
                // 点云相关：腕部/头部 RGBD 彩色或深度都显示实时点云 + D2C 点选
                const pcRelated = handRgbdMode && (
                    (isHand && (isHandDepth || isHandColor)) || isHeadRgbd
                );
                let captureLabel = '采集落盘';
                if (isHeadStereo) captureLabel = '采集落盘（双目）';
                else if (isHand) captureLabel = handRgbdMode ? '采集落盘（深度+彩色）' : '采集落盘（单目）';
                else if (isHeadRgbd) captureLabel = '采集落盘（深度+彩色）';
                const captureBtn = `<button class="btn btn-capture" id="capture-btn-${camera.index}" onclick="captureCamera(${camera.index})" ${projectActive ? '' : 'disabled'}>${captureLabel}</button>`;
                const pcBtns = pcRelated
                    ? `<button class="btn btn-pc-live" id="pc-live-${camera.index}" onclick="toggleLivePointCloud(${camera.index})">实时点云预览</button>
                       <button class="btn btn-pc" onclick="capturePointCloud(${camera.index})">采集点云 (PLY)</button>
                       <button class="btn btn-pc-rec" id="pc-rec-${camera.index}" onclick="togglePcRecord(${camera.index})">连续录点云</button>`
                    : '';
                const d2cBtn = isHeadRgbd
                    ? `<button class="btn btn-d2c-head" id="head-d2c-btn-${camera.index}" onclick="toggleHeadD2CPanel()">深度叠彩图 / 点选测深</button>`
                    : (pcRelated
                        ? `<button class="btn btn-d2c-hand" id="hand-d2c-btn-${camera.index}" onclick="toggleHandD2C(${camera.index})">深度叠彩图 / 点选测深</button>`
                        : '');
                const actions = `<div class="camera-actions">${captureBtn}${pcBtns}${d2cBtn}</div>`;
                card.innerHTML = `
                    <div class="camera-title">${camera.name}</div>
                    <div class="camera-info" id="info-${camera.index}">尺寸: ${camera.shape[0]}x${camera.shape[1]} | 帧率: ${camera.fps.toFixed(1)} FPS</div>
                    <img class="camera-image" id="image-${camera.index}" src="" alt="${camera.name}">
                    ${actions}
                    <div class="capture-msg" id="capture-msg-${camera.index}"></div>
                `;
                grid.appendChild(card);
            });
            refreshPcRecordButtons();
        }

        function updateSelectedImages() {
            const selected = [...selectedIndices];
            if (selected.length === 0 || refreshing) {
                return;
            }
            refreshing = true;
            const ts = Date.now();
            let pending = selected.length;

            const finishOne = () => {
                pending -= 1;
                if (pending <= 0) {
                    refreshing = false;
                    document.getElementById('status').textContent =
                        `预览中: ${selected.length} 路 | ${Date.now() - ts} ms（JPEG直推）`;
                }
            };

            // 每路只请求一次 JPEG（勿再用 Image probe 二次加载，深度伪彩很重）
            selected.forEach((index) => {
                const imgElement = document.getElementById(`image-${index}`);
                const infoElement = document.getElementById(`info-${index}`);
                if (!imgElement) {
                    finishOne();
                    return;
                }
                const url = `/frame/${index}?t=${ts}`;
                const onDone = () => {
                    imgElement.removeEventListener('load', onDone);
                    imgElement.removeEventListener('error', onDone);
                    finishOne();
                };
                const onErr = () => {
                    if (infoElement) {
                        infoElement.textContent = '无法获取图像';
                    }
                    onDone();
                };
                imgElement.addEventListener('load', onDone);
                imgElement.addEventListener('error', onErr);
                imgElement.src = url;
            });
        }

        function captureCamera(index) {
            const btn = document.getElementById(`capture-btn-${index}`);
            const msg = document.getElementById(`capture-msg-${index}`);
            if (!projectActive) {
                if (msg) {
                    msg.style.color = '#b42318';
                    msg.textContent = '请先开始工程';
                }
                return;
            }
            if (btn) btn.disabled = true;
            if (msg) {
                msg.style.color = '#666';
                msg.textContent = '正在采集落盘...';
            }

            fetch('/capture', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: index})
            })
            .then(response => response.json())
            .then(data => {
                if (data.status) renderProjectStatus(data.status);
                if (data.success) {
                    const tip = data.label ? `${data.label} · ` : '';
                    if (msg) {
                        msg.style.color = '#1a7f37';
                        msg.textContent = `${tip}第 ${data.seq} 次已保存到 ${data.save_dir}`;
                    }
                    document.getElementById('status').textContent =
                        `采集成功 #${data.seq} (${data.camera_name || ''}) -> ${data.save_dir}`;
                } else {
                    if (msg) {
                        msg.style.color = '#b42318';
                        msg.textContent = '采集失败: ' + data.error;
                    }
                }
            })
            .catch(error => {
                if (msg) {
                    msg.style.color = '#b42318';
                    msg.textContent = '采集失败: ' + error;
                }
            })
            .finally(() => {
                if (btn) btn.disabled = !projectActive;
            });
        }

        function capturePointCloud(index) {
            const msg = document.getElementById(`capture-msg-${index}`);
            if (msg) {
                msg.style.color = '#666';
                msg.textContent = '正在生成点云...';
            }
            fetch('/capture_pointcloud', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: index, pixel_stride: 1})
            })
            .then(r => r.json())
            .then(data => {
                if (data.status) renderProjectStatus(data.status);
                if (data.success) {
                    let tip = `点云 ${data.point_count} pts → ${data.save_dir}`;
                    if (data.warning) tip += ` | ${data.warning}`;
                    if (msg) {
                        msg.style.color = '#8250df';
                        msg.textContent = tip;
                    }
                    document.getElementById('status').textContent = tip;
                } else if (msg) {
                    msg.style.color = '#b42318';
                    msg.textContent = '点云失败: ' + data.error;
                }
            })
            .catch(err => {
                if (msg) {
                    msg.style.color = '#b42318';
                    msg.textContent = '点云失败: ' + err;
                }
            });
        }

        let pcRecordingIndex = null;
        function refreshPcRecordButtons() {
            fetch('/pointcloud/record/status')
                .then(r => r.json())
                .then(data => {
                    const rec = data.recording || {};
                    pcRecordingIndex = rec.active ? rec.index : null;
                    document.querySelectorAll('[id^="pc-rec-"]').forEach(btn => {
                        const idx = parseInt(btn.id.replace('pc-rec-', ''), 10);
                        if (rec.active && rec.index === idx) {
                            btn.classList.add('active');
                            btn.textContent = `停止录制 (${rec.count || 0})`;
                        } else {
                            btn.classList.remove('active');
                            btn.textContent = '连续录点云';
                        }
                    });
                    if (rec.active) {
                        document.getElementById('status').textContent =
                            `点云录制中 ${rec.side} | 已存 ${rec.count} 帧 → ${rec.dir || ''}`;
                    }
                })
                .catch(() => {});
        }

        function togglePcRecord(index) {
            const msg = document.getElementById(`capture-msg-${index}`);
            const starting = pcRecordingIndex !== index;
            const url = starting ? '/pointcloud/record/start' : '/pointcloud/record/stop';
            const body = starting
                ? JSON.stringify({index: index, interval_s: 0.1, pixel_stride: 2})
                : JSON.stringify({});
            fetch(url, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: body
            })
            .then(r => r.json())
            .then(data => {
                if (msg) {
                    msg.style.color = data.success ? '#bf8700' : '#b42318';
                    msg.textContent = data.message || data.error || JSON.stringify(data);
                }
                refreshPcRecordButtons();
            })
            .catch(err => {
                if (msg) {
                    msg.style.color = '#b42318';
                    msg.textContent = String(err);
                }
            });
        }

        getAvailableCameras();
        // D2C 功能关闭：不自动启动 startHeadD2C()
        const handD2cImg = document.getElementById('hand-d2c-image');
        if (handD2cImg) {
            handD2cImg.addEventListener('click', onHandD2cClick);
        }
        const headD2cImg = document.getElementById('d2c-image');
        if (headD2cImg) {
            headD2cImg.addEventListener('click', onHeadD2cClick);
        }
        setInterval(updateSelectedImages, 200);
        setInterval(refreshProjectStatus, 2000);
        setInterval(refreshPcRecordButtons, 1000);
    </script>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

        let livePc = {
            active: false,
            index: null,
            timer: null,
            busy: false,
            renderer: null,
            scene: null,
            camera: null,
            controls: null,
            points: null,
            geometry: null,
        };

        function ensurePcViewer() {
            const host = document.getElementById('pc-viewer-host');
            if (!host) return false;
            if (livePc.renderer) {
                const w = host.clientWidth || 800;
                const h = host.clientHeight || 460;
                livePc.renderer.setSize(w, h, false);
                livePc.camera.aspect = w / Math.max(1, h);
                livePc.camera.updateProjectionMatrix();
                return true;
            }
            const w = host.clientWidth || 800;
            const h = host.clientHeight || 460;
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0d1117);
            const camera = new THREE.PerspectiveCamera(60, w / Math.max(1, h), 0.01, 20);
            camera.position.set(0, 0, -0.8);
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            renderer.setSize(w, h, false);
            host.innerHTML = '';
            host.appendChild(renderer.domElement);
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.target.set(0, 0, 0.5);
            const grid = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
            grid.rotation.x = Math.PI / 2;
            scene.add(grid);
            const axes = new THREE.AxesHelper(0.2);
            scene.add(axes);
            const geometry = new THREE.BufferGeometry();
            const material = new THREE.PointsMaterial({
                size: 0.004,
                vertexColors: true,
                sizeAttenuation: true,
            });
            const points = new THREE.Points(geometry, material);
            scene.add(points);
            livePc = Object.assign(livePc, {
                renderer, scene, camera, controls, points, geometry,
            });
            function animate() {
                if (!livePc.renderer) return;
                requestAnimationFrame(animate);
                livePc.controls.update();
                livePc.renderer.render(livePc.scene, livePc.camera);
            }
            animate();
            window.addEventListener('resize', () => {
                if (!livePc.active) return;
                ensurePcViewer();
            });
            return true;
        }

        function parseAgpc(buf) {
            const u8 = new Uint8Array(buf);
            if (u8.byteLength < 28) throw new Error('点云数据太短');
            const magic = String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
            if (magic !== 'AGPC') throw new Error('点云格式错误');
            const view = new DataView(buf);
            const count = view.getUint32(8, true);
            let side = '';
            for (let i = 12; i < 28; i++) {
                if (u8[i] === 0) break;
                side += String.fromCharCode(u8[i]);
            }
            const xyzBytes = count * 12;
            const rgbBytes = count * 3;
            const need = 28 + xyzBytes + rgbBytes;
            if (u8.byteLength < need) throw new Error('点云长度不匹配');
            const xyz = new Float32Array(buf, 28, count * 3);
            const rgb = new Uint8Array(buf, 28 + xyzBytes, rgbBytes);
            const colors = new Float32Array(count * 3);
            for (let i = 0; i < count; i++) {
                colors[i * 3] = rgb[i * 3] / 255;
                colors[i * 3 + 1] = rgb[i * 3 + 1] / 255;
                colors[i * 3 + 2] = rgb[i * 3 + 2] / 255;
            }
            return { count, side, xyz: xyz.slice(), colors };
        }

        function updatePcGeometry(parsed) {
            if (!ensurePcViewer()) return;
            livePc.geometry.setAttribute(
                'position',
                new THREE.BufferAttribute(parsed.xyz, 3)
            );
            livePc.geometry.setAttribute(
                'color',
                new THREE.BufferAttribute(parsed.colors, 3)
            );
            livePc.geometry.computeBoundingSphere();
            livePc.geometry.attributes.position.needsUpdate = true;
            livePc.geometry.attributes.color.needsUpdate = true;
        }

        function setLiveButtons(activeIndex) {
            document.querySelectorAll('[id^="pc-live-"]').forEach(btn => {
                const id = parseInt(btn.id.replace('pc-live-', ''), 10);
                const on = activeIndex !== null && id === activeIndex;
                btn.classList.toggle('active', on);
                btn.textContent = on ? '停止点云预览' : '实时点云预览';
            });
        }

        async function fetchLivePointCloud(index) {
            if (livePc.busy || !livePc.active) return;
            livePc.busy = true;
            const meta = document.getElementById('pc-viewer-meta');
            try {
                const r = await fetch(
                    `/pointcloud_preview/${index}?stride=1&max_points=120000`
                );
                if (!r.ok) {
                    const err = await r.json().catch(() => ({}));
                    throw new Error(err.error || (`HTTP ${r.status}`));
                }
                const buf = await r.arrayBuffer();
                const parsed = parseAgpc(buf);
                updatePcGeometry(parsed);
                const warn = r.headers.get('X-PC-Warning') || '';
                const dw = r.headers.get('X-PC-Width') || '640';
                const dh = r.headers.get('X-PC-Height') || '480';
                if (meta) {
                    meta.textContent =
                        `${parsed.side || ''} · ${dw}x${dh} · ${parsed.count} pts · 拖拽旋转 / 滚轮缩放` +
                        (warn === 'hfov-estimate' ? ' · 内参为 FOV 估计' : '');
                }
            } catch (e) {
                if (meta) meta.textContent = '点云预览失败: ' + e.message;
            } finally {
                livePc.busy = false;
            }
        }

        window.toggleLivePointCloud = function(index) {
            if (livePc.active && livePc.index === index) {
                window.stopLivePointCloud();
                return;
            }
            window.stopLivePointCloud();
            livePc.active = true;
            livePc.index = index;
            const panel = document.getElementById('pc-viewer-panel');
            const title = document.getElementById('pc-viewer-title');
            const cam = availableCameras.find(c => c.index === index);
            const isHead = cam && String(cam.name || '').includes('头部RGBD');
            if (panel) panel.classList.add('active');
            if (title) {
                title.textContent = isHead
                    ? `头部点云实时预览 (#${index})`
                    : `手部点云实时预览 (#${index})`;
            }
            setLiveButtons(index);
            ensurePcViewer();
            fetchLivePointCloud(index);
            livePc.timer = setInterval(() => fetchLivePointCloud(index), 200);
            const msg = document.getElementById(`capture-msg-${index}`);
            if (msg) {
                msg.style.color = '#0969da';
                msg.textContent = '实时点云预览中…';
            }
        };

        window.stopLivePointCloud = function() {
            if (livePc.timer) {
                clearInterval(livePc.timer);
                livePc.timer = null;
            }
            livePc.active = false;
            livePc.index = null;
            livePc.busy = false;
            setLiveButtons(null);
            const panel = document.getElementById('pc-viewer-panel');
            if (panel) panel.classList.remove('active');
        };

        // 打开页面自动开实时点云：?live_pc=<index>（默认右手深度；live_pc=off 关闭）
        function maybeAutoStartLivePointCloud() {
            if (!handRgbdMode || livePc.active) return;
            const params = new URLSearchParams(window.location.search || '');
            let raw = params.get('live_pc');
            if (raw === 'off' || raw === 'false' || raw === '0') return;
            let idx;
            if (raw != null && raw !== '') {
                idx = parseInt(raw, 10);
            } else {
                const depths = availableCameras.filter(
                    c => /手|腕/.test(c.name) && c.name.includes('深度'));
                const right = depths.find(c => c.name.includes('右'));
                const pick = right || depths[depths.length - 1] || depths[0];
                if (!pick) return;
                idx = pick.index;
            }
            if (!Number.isFinite(idx)) return;
            if (!selectedIndices.has(idx)) {
                selectedIndices.add(idx);
                syncSelectorChecks();
                createCameraCards();
                updateSelectedImages();
            }
            setTimeout(() => window.toggleLivePointCloud(idx), 300);
        }
    </script>
</body>
</html>
        """
        hand_rgbd_js = 'true' if self.hand_rgbd else 'false'
        capture_hint = (
            '先设置采集时长并「开始工程」→ 在对应相机卡片点「采集落盘」写入同一工程 images/'
            '（头部双目成对；手部深度+彩色成对）；内外参在开工程时落盘一次到 calib/'
            if self.hand_rgbd else
            '先设置采集时长并「开始工程」→ 在对应相机卡片点「采集落盘」写入同一工程 images/'
            '（头部双目成对；手部单目彩色单路）；内外参在开工程时落盘一次到 calib/'
        )
        html_template = (
            html_template
            .replace('/*HAND_RGBD_MODE*/', f'const handRgbdMode = {hand_rgbd_js};')
            .replace('<!--HAND_CAPTURE_HINT-->', capture_hint)
        )
        
        @self.app.route('/')
        def index():
            return render_template_string(html_template)

        @self.app.route('/frame/<int:index>')
        def frame(index):
            """低延迟预览：直接返回 JPEG 二进制（适合 WiFi/远程）"""
            if not (0 <= index < len(self.camera_types)):
                return Response(status=404)
            jpeg = self.get_preview_jpeg_bytes(index)
            if not jpeg:
                return Response(status=204)
            resp = Response(jpeg, mimetype='image/jpeg')
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            resp.headers['Pragma'] = 'no-cache'
            return resp

        @self.app.route('/hand_d2c/<int:index>')
        def hand_d2c(index):
            """手部彩色+D2C深度叠加 JPEG"""
            try:
                alpha = float(request.args.get('alpha', 0.45))
                z_min = float(request.args.get('z_min', 200))
                z_max = float(request.args.get('z_max', 3000))
                mode = request.args.get('mode', 'overlay')
                edges = str(request.args.get('edges', '0')).lower() in (
                    '1', 'true', 'yes', 'on')
            except Exception:
                return Response(status=400)
            result = self.build_hand_d2c_overlay(
                index=index,
                alpha=alpha,
                z_min_mm=z_min,
                z_max_mm=z_max,
                mode=mode,
                edges=edges,
            )
            if not result.get('success'):
                return Response(status=204)
            resp = Response(result['jpeg'], mimetype='image/jpeg')
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            resp.headers['Pragma'] = 'no-cache'
            meta = result.get('meta') or {}
            if meta.get('side'):
                resp.headers['X-Hand-Side'] = str(meta['side'])
            if meta.get('valid_depth_ratio') is not None:
                resp.headers['X-D2C-Valid'] = str(meta['valid_depth_ratio'])
            return resp

        @self.app.route('/hand_d2c/<int:index>/meta')
        def hand_d2c_meta(index):
            result = self.build_hand_d2c_overlay(index=index, mode='overlay')
            if not result.get('success'):
                return jsonify(result)
            return jsonify({'success': True, **(result.get('meta') or {})})

        @self.app.route('/hand_d2c/<int:index>/probe')
        def hand_d2c_probe(index):
            """点击彩色图归一化坐标，返回对齐深度(mm)"""
            try:
                nx = float(request.args.get('nx', -1))
                ny = float(request.args.get('ny', -1))
            except Exception:
                return jsonify({'success': False, 'error': '坐标无效'}), 400
            return jsonify(self.probe_hand_depth_at(index, nx, ny))

        @self.app.route('/d2c/head')
        def d2c_head():
            """头部彩色+深度 D2C 叠加预览 JPEG"""
            try:
                alpha = float(request.args.get('alpha', 0.45))
                z_min = float(request.args.get('z_min', 200))
                z_max = float(request.args.get('z_max', 3000))
                mode = request.args.get('mode', 'overlay')
                edges = str(request.args.get('edges', '0')).lower() in (
                    '1', 'true', 'yes', 'on')
            except Exception:
                return Response(status=400)
            result = self.build_head_d2c_overlay(
                alpha=alpha,
                z_min_mm=z_min,
                z_max_mm=z_max,
                mode=mode,
                edges=edges,
            )
            if not result.get('success'):
                return Response(status=204)
            resp = Response(result['jpeg'], mimetype='image/jpeg')
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            resp.headers['Pragma'] = 'no-cache'
            meta = result.get('meta') or {}
            if meta.get('delta_ts_ms') is not None:
                resp.headers['X-D2C-Delta-Ms'] = str(meta['delta_ts_ms'])
            if meta.get('valid_depth_ratio') is not None:
                resp.headers['X-D2C-Valid'] = str(meta['valid_depth_ratio'])
            return resp

        @self.app.route('/d2c/head/meta')
        def d2c_head_meta():
            return jsonify(self.get_head_d2c_meta())

        @self.app.route('/d2c/head/probe')
        def d2c_head_probe():
            """点击头部彩色图归一化坐标，返回对齐深度(mm)"""
            try:
                nx = float(request.args.get('nx', -1))
                ny = float(request.args.get('ny', -1))
            except Exception:
                return jsonify({'success': False, 'error': '坐标无效'}), 400
            return jsonify(self.probe_head_depth_at(nx, ny))

        @self.app.route('/d2c/head/plane_metric', methods=['GET', 'POST'])
        def d2c_head_plane_metric():
            """PnP 板法向 vs 深度拟合一键测。"""
            body = {}
            if request.method == 'POST':
                body = request.get_json(silent=True) or {}
            src = {}
            src.update(body)
            src.update(request.args.to_dict())
            try:
                cols = int(src.get('cols', 9))
                rows = int(src.get('rows', 6))
                square_mm = float(src.get('square_mm', 25))
                dilate_px = int(src.get('dilate_px', 12))
                z_min = float(src.get('z_min', 200))
                z_max = float(src.get('z_max', 3000))
                stride = int(src.get('stride', 1))
                thresh = float(src.get('ransac_thresh_mm', 8))
                include_vis = str(src.get('vis', '0')).lower() in (
                    '1', 'true', 'yes', 'on')
            except Exception as e:
                return jsonify({'success': False, 'error': f'参数错误: {e}'}), 400
            result = self.run_head_plane_metric(
                cols=cols, rows=rows, square_mm=square_mm,
                dilate_px=dilate_px, z_min_mm=z_min, z_max_mm=z_max,
                pixel_stride=stride, ransac_thresh_mm=thresh,
                include_vis=include_vis,
            )
            return jsonify(result)
        
        @self.app.route('/get_image')
        def get_image():
            try:
                self.update_image()
                if self.current_image is not None:
                    image_b64 = self.image_to_base64(self.current_image)
                    if image_b64:
                        camera_type, camera_name = self.camera_types[self.current_camera_index]
                        info = self.get_camera_info(camera_type)
                        return jsonify({
                            'success': True,
                            'image': image_b64,
                            'info': info
                        })
                
                return jsonify({
                    'success': False,
                    'error': '无法获取图像'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/get_available_cameras')
        def get_available_cameras():
            """获取可用的相机列表"""
            try:
                available_cameras = []
                
                for i, (camera_type, camera_name) in enumerate(self.camera_types):
                    try:
                        # 尝试获取相机信息来检测是否可用
                        shape = self.camera.get_image_shape(camera_type)
                        fps = self.camera.get_image_fps(camera_type)
                        
                        # 如果能获取到信息，说明相机可用
                        available_cameras.append({
                            'index': i,
                            'type': str(camera_type),  # 转换为字符串
                            'name': camera_name,
                            'shape': shape,
                            'fps': fps,
                            'available': True
                        })
                    except Exception as e:
                        # 相机不可用，不添加到列表中
                        print(f"⚠️ 相机 {camera_name} 不可用: {e}")
                        continue
                
                return jsonify({
                    'success': True,
                    'cameras': available_cameras
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })

        def _parse_indices(raw_indices) -> List[int]:
            indices = []
            if raw_indices is None:
                return indices
            if isinstance(raw_indices, str):
                parts = [p.strip() for p in raw_indices.split(',') if p.strip()]
            elif isinstance(raw_indices, list):
                parts = raw_indices
            else:
                parts = [raw_indices]
            for p in parts:
                try:
                    idx = int(p)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(self.camera_types):
                    indices.append(idx)
            # 去重并保持顺序
            seen = set()
            ordered = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    ordered.append(idx)
            return ordered

        @self.app.route('/get_images')
        @self.app.route('/get_all_images')
        def get_images():
            """按选定相机索引取图（兼容旧 /get_all_images）"""
            start_time = time.time()
            try:
                indices = _parse_indices(request.args.get('indices'))
                if not indices:
                    # 未指定时兼容旧行为：全部相机
                    indices = list(range(len(self.camera_types)))

                images = {}
                with ThreadPoolExecutor(max_workers=min(4, max(1, len(indices)))) as executor:
                    future_to_index = {
                        executor.submit(
                            self.process_single_camera,
                            self.camera_types[i][0],
                            self.camera_types[i][1],
                            i
                        ): i
                        for i in indices
                    }
                    for future in as_completed(future_to_index):
                        result = future.result()
                        images[result['index']] = result

                processing_time = (time.time() - start_time) * 1000
                cached_count = sum(1 for img in images.values() if img and img.get('cached', False))
                new_frame_count = sum(
                    1 for img in images.values()
                    if img and not img.get('cached', False) and img.get('success', False)
                )
                error_count = sum(1 for img in images.values() if img and not img.get('success', False))
                print(
                    f"📊 处理耗时: {processing_time:.1f}ms | 选中: {len(indices)} | "
                    f"新帧: {new_frame_count} | 缓存: {cached_count} | 错误: {error_count}"
                )

                return jsonify({
                    'success': True,
                    'images': images,
                    'processing_time_ms': round(processing_time, 1),
                    'cached_count': cached_count
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })

        @self.app.route('/project/status')
        def project_status():
            try:
                status = self.get_project_status()
                status['success'] = True
                return jsonify(status)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/project/start', methods=['POST'])
        def project_start():
            try:
                data = request.get_json(silent=True) or {}
                result = self.start_capture_project(
                    duration_sec=data.get('duration_sec'),
                    name=data.get('name'),
                )
                return jsonify(result)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/project/stop', methods=['POST'])
        def project_stop():
            try:
                result = self.stop_capture_project()
                return jsonify(result)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/capture', methods=['POST'])
        def capture():
            """在当前采集工程内落盘：按相机卡片落盘头部双目或手部单目彩色等"""
            try:
                data = request.get_json(silent=True) or {}
                index = data.get('index', None)
                if index is not None:
                    index = int(index)
                result = self.capture_stereo_pair_to_disk(index=index)
                return jsonify(result)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/capture_pointcloud', methods=['POST'])
        def capture_pointcloud():
            """手部 RGBD 点云单帧落盘（PLY + NPY）"""
            try:
                data = request.get_json(silent=True) or {}
                index = int(data.get('index', -1))
                stride = int(data.get('pixel_stride', 1))
                result = self.capture_hand_pointcloud_to_disk(
                    index=index, pixel_stride=stride)
                return jsonify(result)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/pointcloud_preview/<int:index>')
        def pointcloud_preview(index):
            """浏览器实时点云：二进制 AGPC（默认 640x480 / stride=1）"""
            try:
                stride = int(request.args.get('stride', 1))
                max_points = int(request.args.get('max_points', 120000))
                packed = self.pack_pointcloud_preview_bytes(
                    index=index,
                    pixel_stride=max(1, stride),
                    max_points=max(1000, max_points),
                )
                if not packed.get('success'):
                    return jsonify(packed), 400
                resp = Response(packed['payload'], mimetype='application/octet-stream')
                resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
                resp.headers['X-PC-Count'] = str(packed['count'])
                resp.headers['X-PC-Side'] = packed.get('side') or ''
                ds = packed.get('depth_size') or list(self.hand_pc_depth_size)
                resp.headers['X-PC-Width'] = str(int(ds[0]))
                resp.headers['X-PC-Height'] = str(int(ds[1]))
                # HTTP 头必须 latin-1：中文警告放进可 ASCII 的标记
                if packed.get('warning'):
                    resp.headers['X-PC-Warning'] = 'hfov-estimate'
                return resp
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/pointcloud/record/start', methods=['POST'])
        def pointcloud_record_start():
            try:
                data = request.get_json(silent=True) or {}
                result = self.start_hand_pointcloud_recording(
                    index=int(data.get('index', -1)),
                    interval_s=float(data.get('interval_s', 0.1)),
                    pixel_stride=int(data.get('pixel_stride', 2)),
                )
                return jsonify(result)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/pointcloud/record/stop', methods=['POST'])
        def pointcloud_record_stop():
            try:
                return jsonify(self.stop_hand_pointcloud_recording())
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/pointcloud/record/status')
        def pointcloud_record_status():
            try:
                return jsonify({
                    'success': True,
                    'recording': self.get_pointcloud_record_status(),
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/switch_camera', methods=['POST'])
        def switch_camera():
            try:
                data = request.get_json()
                index = data.get('index', 0)
                if 0 <= index < len(self.camera_types):
                    self.current_camera_index = index
                    # 立即更新图像
                    self.update_image()
                    camera_type, camera_name = self.camera_types[index]
                    info = self.get_camera_info(camera_type)
                    return jsonify({
                        'success': True, 
                        'camera_name': camera_name,
                        'info': info
                    })
                return jsonify({'success': False, 'error': '无效的相机索引'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    
    def run(self):
        """运行Web服务器"""
        if not HAS_FLASK:
            print("❌ Flask未安装，无法启动Web服务器")
            print("请运行: pip install flask")
            return
        
        if self.camera is None:
            print("❌ 相机未初始化，无法启动")
            return
        
        print("🌐 Web版相机查看器启动 - 按时长工程采集")
        print("=" * 60)
        print("📷 支持的相机类型:")
        for i, (_, name) in enumerate(self.camera_types):
            print(f"  {i}: {name}")
        print(f"\n💾 采集根目录: {self.save_dir}")
        print(f"⏱ 默认采集时长: {self.default_capture_duration_sec}s")
        print(f"🌐 访问地址: http://{self.host}:{self.port}")
        print("💡 设置采集时长 → 开始工程 → 多次采集落盘到同一工程 images/")
        print("=" * 60)
        
        try:
            self.is_running = True
            self.app.run(host=self.host, port=self.port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断程序")
        except Exception as e:
            print(f"❌ Web服务器启动失败: {e}")
        finally:
            # 只有在信号处理器没有清理的情况下才清理
            if not hasattr(self, '_cleanup_done') or not self._cleanup_done:
                self.cleanup()

def main():
    """主函数"""
    print("🤖 Web版机器人相机查看器")
    print("=" * 50)
    
    # 检查Flask
    if HAS_FLASK:
        print("✅ Flask已安装")
    else:
        print("❌ Flask未安装")
        print("请运行: pip install flask")
        return
    
    # 检查OpenCV
    if HAS_OPENCV:
        print(f"✅ OpenCV版本: {cv2.__version__}")
    else:
        print("⚠️ OpenCV未安装，将使用原始数据模式")
    
    # 检查agibot_gdk
    try:
        import agibot_gdk
        print("✅ agibot_gdk已导入")
    except ImportError:
        print("❌ 错误: agibot_gdk未安装或未正确配置")
        return
    
    # 创建并运行Web查看器
    viewer = WebCameraViewer()
    viewer.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
    finally:
        print("👋 程序已退出")
        # 强制退出，避免任何残留进程
        os._exit(0)
