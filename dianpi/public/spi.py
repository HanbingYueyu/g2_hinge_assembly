#-*- coding: utf-8 -*-
from ctypes import *
from librockmong import *

# 初始化SPI
# SerialNumber: 设备序号
# Channel：通道编号。0，SPI0. 1, SPI1...
# Mode：工作模式。0，主机
# BaudRate：通信速率，单位 Hz
# CPHA：时钟相位。0 或 1
# CPOL：时钟极性。0 或 1
# SelPolarity：片选极性。0，低电平有效。1，高电平有效
# 函数返回：0，正常；<0，异常
def SPI_Init(SerialNumber, Channel, Mode, BaudRate, CPHA, CPOL, SelPolarity):
    return librockmong.SPI_Init(SerialNumber, Channel, Mode, BaudRate, CPHA, CPOL, SelPolarity)

# SPI写数据
# SerialNumber: 设备序号
# Channel：通道编号
# WriteBuffer：写入数据，按 int 数组传入，每个元素取低 8 位
# WriteLength：写入字节数
# 函数返回：0，正常；<0，异常
def SPI_WriteBytes(SerialNumber, Channel, WriteBuffer, WriteLength):
    return librockmong.SPI_WriteBytes(SerialNumber, Channel, WriteBuffer, WriteLength)

# SPI读数据
# SerialNumber: 设备序号
# Channel：通道编号
# ReadBuffer：读取缓冲区，按 int 数组传入
# ReadLength：读取字节数
# 函数返回：0，正常；<0，异常
def SPI_ReadBytes(SerialNumber, Channel, ReadBuffer, ReadLength):
    return librockmong.SPI_ReadBytes(SerialNumber, Channel, ReadBuffer, ReadLength)

# SPI同时写和读
# SerialNumber: 设备序号
# Channel：通道编号
# WriteBuffer：写入数据
# ReadBuffer：读取缓冲区
# WriteLength：写入字节数
# ReadLength：读取字节数
# 函数返回：0，正常；<0，异常
def SPI_WriteReadBytes(SerialNumber, Channel, WriteBuffer, ReadBuffer, WriteLength, ReadLength):
    return librockmong.SPI_WriteReadBytes(SerialNumber, Channel, WriteBuffer, ReadBuffer, WriteLength, ReadLength)
