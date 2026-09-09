#-*- coding: utf-8 -*-
from ctypes import *
from librockmong import *

# 擦除用户数据区。写入前需要先擦除
# SerialNumber: 设备序号（SN）
# 函数返回：0，正常；<0，异常
def UserData_Erase(SerialNumber):
    return librockmong.UserData_Erase(SerialNumber)

# 向用户数据区写入数据
# SerialNumber: 设备序号（SN）
# Address：起始地址
# Size：字节数
# Dat：要写入的数据
# 函数返回：0，正常；<0，异常
def UserData_Write(SerialNumber, Address, Size, Dat):
    return librockmong.UserData_Write(SerialNumber, Address, Size, Dat)

# 从用户数据区读取数据
# SerialNumber: 设备序号（SN）
# Address：起始地址
# Size：字节数
# Dat：读取缓冲区
# 函数返回：0，正常；<0，异常
def UserData_Read(SerialNumber, Address, Size, Dat):
    return librockmong.UserData_Read(SerialNumber, Address, Size, Dat)
