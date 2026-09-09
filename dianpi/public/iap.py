#-*- coding: utf-8 -*-
from ctypes import *
import platform
from librockmong import *

# 跳转到 Boot 升级模式。调用后设备会重启进入 IAP
# SerialNumber: 设备序号（SN）
# 函数返回：0，正常；<0，异常
def iap_jump_boot(SerialNumber):
    return librockmong.iap_jump_boot(SerialNumber)

# 向 App 区写入固件数据
# SerialNumber: 设备序号（SN）
# address：写入起始地址
# data：固件数据
# length：字节数
# 函数返回：0，正常；<0，异常
def iap_write_app(SerialNumber, address, data, length):
    return librockmong.iap_write_app(SerialNumber, address, data, length)

# 写完固件后跳转到 App 运行
# SerialNumber: 设备序号（SN）
# 函数返回：0，正常；<0，异常
def iap_jump_app(SerialNumber):
    return librockmong.iap_jump_app(SerialNumber)
