"""Read-only Linux hidraw input for SpaceMouse Compact (256f:c635).

Report layout: https://github.com/JakubAndrysek/PySpaceMouse/blob/master/pyspacemouse/devices.toml
Keep raw XYZ/RXYZ ordering here; the teleop mapping defines the robot frame.
No daemon, HID output reports, or third-party driver is required.
"""

from __future__ import annotations

import fcntl
import math
import os
from pathlib import Path
import struct
import time
from dataclasses import dataclass


class InputError(RuntimeError):
    pass


@dataclass(frozen=True)
class InputFrame:
    axes: tuple[float, ...]
    buttons: tuple[bool, bool]
    pressed: tuple[int, ...]
    axis_times: tuple[float | None, float | None]
    ready: bool
    peak_axes: tuple[float, ...] = (0.0,) * 6


def parse_axis_map(text: str) -> tuple[int, ...]:
    """Signed 1-based raw axis indices, translation first, then rotation."""
    values = tuple(int(part) for part in text.split(','))
    if len(values) != 6 or set(map(abs, values[:3])) != {1, 2, 3} or set(map(abs, values[3:])) != {4, 5, 6}:
        raise ValueError('轴映射必须是 1,2,3 和 4,5,6 各自的带符号排列')
    return values


class CompactReports:
    def __init__(self) -> None:
        self._axes = [0.0] * 6
        self._times: list[float | None] = [None, None]
        self._buttons = (False, False)
        self._buttons_seen = False
        self._pressed: list[int] = []
        self._peak_axes = [0.0] * 6

    def feed(self, data: bytes, now: float) -> None:
        if not data or not math.isfinite(now):
            raise InputError('无效 HID 报告或时间戳')
        channel = data[0]
        if channel in (1, 2):
            if len(data) != 7:
                raise InputError(f'Compact 轴报告长度错误: {len(data)}，预期 7')
            offset = (channel - 1) * 3
            self._axes[offset:offset + 3] = [max(-1.0, min(1.0, value / 350.0)) for value in struct.unpack('<hhh', data[1:])]
            for index in range(offset, offset + 3):
                self._peak_axes[index] = max(self._peak_axes[index], abs(self._axes[index]))
            self._times[channel - 1] = now
        elif channel == 3:
            if len(data) < 2:
                raise InputError('Compact 按钮报告过短')
            buttons = (bool(data[1] & 1), bool(data[1] & 2))
            if self._buttons_seen:
                for index in range(2):
                    if buttons[index] and not self._buttons[index]:
                        self._pressed.append(index)
            self._buttons = buttons
            self._buttons_seen = True

    def snapshot(self) -> InputFrame:
        frame = InputFrame(tuple(self._axes), self._buttons, tuple(self._pressed), tuple(self._times),
                           self._buttons_seen and all(stamp is not None for stamp in self._times), tuple(self._peak_axes))
        self._pressed.clear()
        self._peak_axes = [0.0] * 6
        return frame


def discover_devices(sys_root: Path = Path('/sys/class/hidraw')) -> list[str]:
    found = []
    for entry in sorted(sys_root.glob('hidraw*')):
        try:
            fields = dict(line.split('=', 1) for line in (entry / 'device/uevent').read_text().splitlines() if '=' in line)
            bus, vendor, product = (int(part, 16) for part in fields.get('HID_ID', '').split(':'))
        except (OSError, ValueError):
            continue
        if bus == 3 and (vendor, product) == (0x256F, 0xC635):
            found.append('/dev/' + entry.name)
    return found


def resolve_discovered_device(path: str, devices: list[str]) -> str | None:
    """Return the discovered hidraw node addressed by a direct path or symlink."""
    requested = os.path.realpath(path)
    for device in devices:
        if requested == os.path.realpath(device):
            return device
    return None


class CompactHID:
    """Nonblocking reader. Only successful reads advance report timestamps."""

    def __init__(self, path: str | None = None) -> None:
        devices = discover_devices()
        if path is None:
            if len(devices) != 1:
                raise InputError(f'找到 {len(devices)} 个 SpaceMouse Compact；使用 --list-devices / --device 指定')
            path = devices[0]
        if resolve_discovered_device(path, devices) is None:
            raise InputError(f'{path} 不是已识别的 USB SpaceMouse Compact (256f:c635)')
        self.path = path
        self._fd: int | None = None
        self._reports = CompactReports()
        try:
            self._fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except PermissionError as exc:
            self.close()
            raise InputError(f'无权读取 {path}；参见 g2_adapter/SPACEMOUSE.md 的 udev 权限设置') from exc
        except OSError:
            self.close()
            raise

    def poll(self) -> InputFrame:
        if self._fd is None:
            raise InputError('SpaceMouse 已关闭')
        # Drain queued reports, including short button clicks. An unbounded
        # drain could starve the 50-Hz writer; a backlog fails closed instead.
        for _ in range(128):
            try:
                data = os.read(self._fd, 64)
            except BlockingIOError:
                return self._reports.snapshot()
            except OSError as exc:
                raise InputError(f'SpaceMouse 读取失败/已断连: {exc}') from exc
            if not data:
                raise InputError('SpaceMouse 已断连')
            self._reports.feed(data, time.monotonic())
        raise InputError('SpaceMouse 输入积压，停止以避免执行历史输入')

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
