#-*- coding: utf-8 -*-
from ctypes import *
import platform
from librockmong import *

# 扫描网络设备。只找网口设备，不含 USB/串口。会清空当前网络设备列表
# SerialNumbers：输出数组，存放扫描到的设备 SN。调用前请分配足够空间，建议不少于 16 个 int
# 函数返回：>0，设备个数；0，未找到设备；<0，异常
def NetDevice_Scan(SerialNumbers):
    return librockmong.NetDevice_Scan(SerialNumbers)

# 打开网络设备。打开后才能调用 IO/PWM/ADC 等功能接口
# SerialNumber: 设备序号（SN），来自 NetDevice_Scan 或 NetDevice_AddWithIp
# 函数返回：0，正常；<0，异常
def NetDevice_Open(SerialNumber):
    return librockmong.NetDevice_Open(SerialNumber)

# 关闭网络设备
# SerialNumber: 设备序号（SN）
# 函数返回：0，正常；<0，异常
def NetDevice_Close(SerialNumber):
    return librockmong.NetDevice_Close(SerialNumber)

# 按 IPv4 添加一台网络设备，不扫描。已存在相同 SN 时复用并更新地址
# Address：IPv4 字符串，如 192.168.1.10
# SerialNumber：c_int，成功后写入该设备 SN
# 函数返回：0，正常；<0，异常
def NetDevice_AddWithIp(Address, SerialNumber):
    if isinstance(Address, str):
        Address = Address.encode("utf-8")
    return librockmong.NetDevice_AddWithIp(Address, SerialNumber)
