#-*- coding: utf-8 -*-
from ctypes import *
from librockmong import *

# 初始化设备上的 UART 通道（设备对外串口，不是电脑侧 COM）
# SerialNumber: 设备序号
# Channel：通道编号。0，UART0. 1, UART1...
# BaudRate：波特率，单位 bps，如 115200
# WordLength：数据位。一般填 8
# StopBits：停止位。一般填 1
# Parity：校验。0，无。1，奇。2，偶
# HardwareFlowCtrl：硬件流控。0，无。1，RTS/CTS
# 函数返回：0，正常；<0，异常
def UART_Init(SerialNumber, Channel, BaudRate, WordLength, StopBits, Parity, HardwareFlowCtrl):
    return librockmong.UART_Init(SerialNumber, Channel, BaudRate, WordLength, StopBits, Parity, HardwareFlowCtrl)

# 向 UART 写数据
# SerialNumber: 设备序号
# Channel：通道编号
# Buffer：要发送的数据
# Length：字节数
# 函数返回：0，正常；<0，异常
def UART_Write(SerialNumber, Channel, Buffer, Length):
    return librockmong.UART_Write(SerialNumber, Channel, Buffer, Length)

# 从 UART 读数据
# SerialNumber: 设备序号
# Channel：通道编号
# Buffer：读取缓冲区
# ActualLength：c_int，返回实际读到的字节数
# MaxLength：缓冲区最大字节数
# TimeoutMs：超时时间，单位 ms
# 函数返回：0，正常；<0，异常
def UART_Read(SerialNumber, Channel, Buffer, ActualLength, MaxLength, TimeoutMs):
    return librockmong.UART_Read(SerialNumber, Channel, Buffer, ActualLength, MaxLength, TimeoutMs)

# 清空 UART 收发缓冲
# SerialNumber: 设备序号
# Channel：通道编号
# 函数返回：0，正常；<0，异常
def UART_Clear(SerialNumber, Channel):
    return librockmong.UART_Clear(SerialNumber, Channel)
