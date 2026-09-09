#-*- coding: utf-8 -*-
from ctypes import *
import platform
from librockmong import *

# 扫描已连接的设备（USB、网口、串口）。会清空当前设备列表再重新扫描
# 若感觉 Device_Scan 太慢，可按需改用 UsbDevice_Scan、NetDevice_Scan、CH343Dev_Scan
# SerialNumbers：输出数组，存放扫描到的设备 SN。调用前请分配足够空间，建议不少于 16 个 int
# 函数返回：>0，设备个数；0，未找到设备；<0，异常
def Device_Scan(SerialNumbers):
    return librockmong.Device_Scan(SerialNumbers)

# 按地址添加一台设备，不扫描。已存在相同 SN 时复用并更新地址
# Address：串口名（如 COM5、/dev/ttyUSB0）或 IPv4（如 192.168.1.10）。USB 请用 Device_Scan
# SerialNumber：输出，成功后写入该设备 SN
# 函数返回：0，正常；<0，异常
def Device_Add(Address, SerialNumber):
    if isinstance(Address, str):
        Address = Address.encode("utf-8")
    return librockmong.Device_Add(Address, SerialNumber)

# 打开设备。打开后才能调用 IO/PWM/ADC 等功能接口
# SerialNumber: 设备序号（SN），来自 Device_Scan 或 Device_Add
# 函数返回：0，正常；<0，异常
def Device_Open(SerialNumber):
    return librockmong.Device_Open(SerialNumber)

# 关闭设备
# SerialNumber: 设备序号（SN）
# 函数返回：0，正常；<0，异常
def Device_Close(SerialNumber):
    return librockmong.Device_Close(SerialNumber)

# 读取设备类型
# SerialNumber: 设备序号（SN）
# Type：返回设备类型。1，USB。2，网络。4，CH343 串口。8，普通串口
# 函数返回：0，正常；<0，异常
def Device_GetType(SerialNumber, Type):
    return librockmong.Device_GetType(SerialNumber, Type)

# 读取设备 IP。仅网络设备有效，USB/串口设备会失败
# SerialNumber: 设备序号（SN）
# IP：输出缓冲区，写入以 '\\0' 结尾的 IPv4 字符串，建议不少于 16 字节
# 函数返回：0，正常；<0，异常
def Device_GetIP(SerialNumber, IP):
    return librockmong.Device_GetIP(SerialNumber, IP)
