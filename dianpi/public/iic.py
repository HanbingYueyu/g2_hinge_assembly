#-*- coding: utf-8 -*-
from ctypes import *
from librockmong import *

# 初始化I2C
# SerialNumber: 设备序号
# Channel：通道编号。0，I2C0. 1, I2C1...
# Mode：工作模式。0，主机
# AddressLength：从机地址位数。7 或 10
# OwnAddress：本机地址。主机模式一般填 0
# Frequency：I2C 时钟频率，单位 Hz
# PullUp：上拉。0，无。1，使能内部上拉
# 函数返回：0，正常；<0，异常
def IIC_Init(SerialNumber, Channel, Mode, AddressLength, OwnAddress, Frequency, PullUp):
    return librockmong.IIC_Init(SerialNumber, Channel, Mode, AddressLength, OwnAddress, Frequency, PullUp)

# 扫描总线上的从机地址
# SerialNumber: 设备序号
# Channel：通道编号
# AddressLength：从机地址位数。7 或 10
# SlaveAddress：输出数组，存放扫描到的从机地址。调用前请分配足够空间
# 函数返回：>0，扫描到的从机个数；<0，异常
def IIC_GetSlaveAddress(SerialNumber, Channel, AddressLength, SlaveAddress):
    return librockmong.IIC_GetSlaveAddress(SerialNumber, Channel, AddressLength, SlaveAddress)

# 向从机写数据
# SerialNumber: 设备序号
# Channel：通道编号
# SlaveAddress：从机地址
# WriteBuffer：写入数据，按 int 数组传入，每个元素取低 8 位
# WriteCount：写入字节数
# Timeout：超时时间，单位 ms
# 函数返回：0，正常；<0，异常
def IIC_WriteBytes(SerialNumber, Channel, SlaveAddress, WriteBuffer, WriteCount, Timeout):
    return librockmong.IIC_WriteBytes(SerialNumber, Channel, SlaveAddress, WriteBuffer, WriteCount, Timeout)

# 从从机读数据
# SerialNumber: 设备序号
# Channel：通道编号
# SlaveAddress：从机地址
# Buffer：读取缓冲区，按 int 数组传入
# ReadCount：读取字节数
# Timeout：超时时间，单位 ms
# 函数返回：0，正常；<0，异常
def IIC_ReadBytes(SerialNumber, Channel, SlaveAddress, Buffer, ReadCount, Timeout):
    return librockmong.IIC_ReadBytes(SerialNumber, Channel, SlaveAddress, Buffer, ReadCount, Timeout)

# 先写后读（带重复起始）
# SerialNumber: 设备序号
# Channel：通道编号
# SlaveAddress：从机地址
# WriteBuffer：先写入的数据
# WriteCount：写入字节数
# ReadBuffer：随后读出的数据
# ReadCount：读取字节数
# Timeout：超时时间，单位 ms
# 函数返回：0，正常；<0，异常
def IIC_WriteReadBytes(SerialNumber, Channel, SlaveAddress, WriteBuffer, WriteCount, ReadBuffer, ReadCount, Timeout):
    return librockmong.IIC_WriteReadBytes(SerialNumber, Channel, SlaveAddress, WriteBuffer, WriteCount, ReadBuffer, ReadCount, Timeout)
