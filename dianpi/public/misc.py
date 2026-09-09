#-*- coding: utf-8 -*-
from ctypes import *
import platform
from librockmong import *

# 读取设备固件版本
# SerialNumber: 设备序号（SN）
# Ver：输出缓冲区，返回版本字符串，建议不少于 32 字节
# 函数返回：0，正常；<0，异常
def MISC_GetFwVersion(SerialNumber, Version):
    return librockmong.MISC_GetFwVersion(SerialNumber, Version)

# 读取设备型号
# SerialNumber: 设备序号（SN）
# Model：输出缓冲区，返回型号字符串，建议不少于 32 字节
# 函数返回：0，正常；<0，异常
def MISC_GetModel(SerialNumber, Model):
    return librockmong.MISC_GetModel(SerialNumber, Model)
	
# 读取电脑端库版本（不需要打开设备）
# Ver：输出缓冲区，返回版本字符串，建议不少于 32 字节
# 函数返回：0，正常；<0，异常
def MISC_GetLibraryVersion(Version):
    return librockmong.MISC_GetLibraryVersion(Version)
	
# 修改设备序列号
# SerialNumber: 当前设备序号（SN）
# NewSerialNumber：要写成的新 SN
# 函数返回：0，正常；<0，异常
def MISC_SetSn(SerialNumber, NewSerialNumber):
    return librockmong.MISC_SetSn(SerialNumber, NewSerialNumber)

# 读取设备内部数据
# SerialNumber: 设备序号（SN）
# Data：输出缓冲区
# Length：要读取的字节数
# 函数返回：0，正常；<0，异常
def MISC_GetData(SerialNumber, Data, Length):
    return librockmong.MISC_GetData(SerialNumber, Data, Length)

# 读取 USB 序列号字符串
# SerialNumber: 设备序号（SN）
# UsbSerialNumber：输出缓冲区，建议不少于 32 字节
# 函数返回：0，正常；<0，异常
def MISC_GetUsbSerialNumber(SerialNumber, UsbSerialNumber):
    return librockmong.MISC_GetUsbSerialNumber(SerialNumber, UsbSerialNumber)

# 设置端口电平（电源电压）
# SerialNumber: 设备序号
# Millivolt：电压，单位 mV
# 函数返回：0，正常；<0，异常
def MISC_SetPortLevel(SerialNumber, Millivolt):
    return librockmong.MISC_SetPortLevel(SerialNumber, Millivolt)

# 读取端口电平（电源电压）
# SerialNumber: 设备序号
# Millivolt：c_int，返回电压，单位 mV
# 函数返回：0，正常；<0，异常
def MISC_GetPortLevel(SerialNumber, Millivolt):
    return librockmong.MISC_GetPortLevel(SerialNumber, Millivolt)

# 设置设备 IP（仅网络设备）
# SerialNumber: 设备序号（SN）
# IP：IPv4 字符串，如 192.168.1.10
# 函数返回：0，正常；<0，异常
def MISC_SetIP(SerialNumber, IP):
    if isinstance(IP, str):
        IP = IP.encode("utf-8")
    return librockmong.MISC_SetIP(SerialNumber, IP)

# 读取设备 IP（仅网络设备）
# SerialNumber: 设备序号（SN）
# IP：输出缓冲区，建议不少于 16 字节
# 函数返回：0，正常；<0，异常
def MISC_GetIP(SerialNumber, IP):
    return librockmong.MISC_GetIP(SerialNumber, IP)

# 复位设备
# SerialNumber: 设备序号（SN）
# 函数返回：0，正常；<0，异常
def MISC_Reset(SerialNumber):
    return librockmong.MISC_Reset(SerialNumber)

	