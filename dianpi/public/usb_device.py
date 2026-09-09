#-*- coding: utf-8 -*-
from ctypes import *
import platform
from librockmong import *

# 扫描 USB 设备。只找 USB，不含网口和串口。会清空当前 USB 设备列表
# SerialNumbers：输出数组，存放扫描到的设备 SN。调用前请分配足够空间，建议不少于 16 个 int
# 函数返回：>0，设备个数；0，未插入设备；<0，异常
def UsbDevice_Scan(SerialNumbers):
    return librockmong.UsbDevice_Scan(SerialNumbers)

# 打开 USB 设备。打开后才能调用 IO/PWM/ADC 等功能接口
# SerialNumber: 设备序号（SN），来自 UsbDevice_Scan
# 函数返回：0，正常；<0，异常
def UsbDevice_Open(SerialNumber):
    return librockmong.UsbDevice_Open(SerialNumber)

# 关闭 USB 设备
# SerialNumber: 设备序号（SN）
# 函数返回：0，正常；<0，异常
def UsbDevice_Close(SerialNumber):
    return librockmong.UsbDevice_Close(SerialNumber)
