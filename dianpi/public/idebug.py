#-*- coding: utf-8 -*-
from ctypes import *
import platform
from librockmong import *

IDEBUG_LOG_NONE = 0
IDEBUG_LOG_ERROR = 1
IDEBUG_LOG_WARNING = 2
IDEBUG_LOG_INFO = 3
IDEBUG_LOG_DEBUG = 4

if platform.system() == "Windows":
    _LogCb = WINFUNCTYPE(None, c_int, c_char_p)
else:
    _LogCb = CFUNCTYPE(None, c_int, c_char_p)

_log_cb_ref = None

# 打开或关闭调试日志
# enable：1，相当于级别 DEBUG。0，关闭日志
# 无返回值
def idebug_enable(enable):
    librockmong.idebug_enable(enable)

# 设置日志级别
# level：取值 IDEBUG_LOG_NONE ~ IDEBUG_LOG_DEBUG
# 无返回值
def idebug_set_log_level(level):
    librockmong.idebug_set_log_level(level)

# 读取当前日志级别
# 函数返回：当前级别，取值 IDEBUG_LOG_NONE ~ IDEBUG_LOG_DEBUG
def idebug_get_log_level():
    return librockmong.idebug_get_log_level()

# 设置日志回调。传 None 取消回调，改回默认输出（桌面 stderr / Android logcat）
# cb：回调函数，参数为 (level, msg)。msg 为 UTF-8 文本
# 无返回值
def idebug_set_log_cb(cb):
    global _log_cb_ref
    if cb is None:
        _log_cb_ref = None
        librockmong.idebug_set_log_cb(None)
        return

    def _wrap(level, msg):
        text = msg.decode("utf-8", "replace") if isinstance(msg, bytes) else msg
        cb(level, text)

    _log_cb_ref = _LogCb(_wrap)
    librockmong.idebug_set_log_cb(_log_cb_ref)
