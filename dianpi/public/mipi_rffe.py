#-*- coding: utf-8 -*-
from ctypes import *
from librockmong import *

# 初始化 MIPI RFFE 通道
# SerialNumber: 设备序号
# Channel：通道编号。0，RFFE0. 1, RFFE1...
# Frequency：通信频率，单位 Hz
# 函数返回：0，正常；<0，异常
def RFFE_Init(SerialNumber, Channel, Frequency):
    return librockmong.RFFE_Init(SerialNumber, Channel, Frequency)

# 写寄存器 0（RFFE 命令：Register 0 Write）
# SerialNumber: 设备序号
# Channel：通道编号
# SlaveAddress：从机地址
# Value：写入值，0~255
# 函数返回：0，正常；<0，异常
def RFFE_Reg0_Write(SerialNumber, Channel, SlaveAddress, Value):
    return librockmong.RFFE_Reg0_Write(SerialNumber, Channel, SlaveAddress, Value)

# 写普通寄存器
# SerialNumber: 设备序号
# Channel：通道编号
# SlaveAddress：从机地址
# RegisterAddress：寄存器地址
# Value：写入值，0~255
# 函数返回：0，正常；<0，异常
def RFFE_Reg_Write(SerialNumber, Channel, SlaveAddress, RegisterAddress, Value):
    return librockmong.RFFE_Reg_Write(SerialNumber, Channel, SlaveAddress, RegisterAddress, Value)

# 读普通寄存器
# SerialNumber: 设备序号
# Channel：通道编号
# SlaveAddress：从机地址
# RegisterAddress：寄存器地址
# Value：返回读取值
# 函数返回：0，正常；<0，异常
def RFFE_Reg_Read(SerialNumber, Channel, SlaveAddress, RegisterAddress, Value):
    return librockmong.RFFE_Reg_Read(SerialNumber, Channel, SlaveAddress, RegisterAddress, Value)

# 写扩展寄存器（连续写多个字节）
# SerialNumber: 设备序号
# Channel：通道编号
# SlaveAddress：从机地址
# RegisterAddress：起始寄存器地址
# WriteBuffer：写入数据，按 int 数组传入，每个元素取低 8 位
# WriteCount：写入字节数
# 函数返回：0，正常；<0，异常
def RFFE_Ext_Reg_Write(SerialNumber, Channel, SlaveAddress, RegisterAddress, WriteBuffer, WriteCount):
    return librockmong.RFFE_Ext_Reg_Write(SerialNumber, Channel, SlaveAddress, RegisterAddress, WriteBuffer, WriteCount)

# 读扩展寄存器（连续读多个字节）
# SerialNumber: 设备序号
# Channel：通道编号
# SlaveAddress：从机地址
# RegisterAddress：起始寄存器地址
# ReadBuffer：读取缓冲区，按 int 数组传入
# ReadCount：读取字节数
# 函数返回：0，正常；<0，异常
def RFFE_Ext_Reg_Read(SerialNumber, Channel, SlaveAddress, RegisterAddress, ReadBuffer, ReadCount):
    return librockmong.RFFE_Ext_Reg_Read(SerialNumber, Channel, SlaveAddress, RegisterAddress, ReadBuffer, ReadCount)
