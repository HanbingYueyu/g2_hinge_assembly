#-*- coding: utf-8 -*-
from ctypes import *
from librockmong import *

# 初始化DAC，下载波形数据。不会立即输出，需再调用 DAC_Start
# SerialNumber: 设备序号
# Channel：通道编号。0，DAC0. 1, DAC1...
# IntervalNs：相邻两点的时间间隔，单位 ns
# Buffer：波形数据数组。按 int 传入，库内转成 16 位
# Size：波形点数
# 函数返回：0，正常；<0，异常
def DAC_Init(SerialNumber, Channel, IntervalNs, Buffer, Size):
    return librockmong.DAC_Init(SerialNumber, Channel, IntervalNs, Buffer, Size)

# 开始DAC输出
# SerialNumber: 设备序号
# Channel：通道编号
# 函数返回：0，正常；<0，异常
def DAC_Start(SerialNumber, Channel):
    return librockmong.DAC_Start(SerialNumber, Channel)

# 停止DAC输出
# SerialNumber: 设备序号
# Channel：通道编号
# 函数返回：0，正常；<0，异常
def DAC_Stop(SerialNumber, Channel):
    return librockmong.DAC_Stop(SerialNumber, Channel)
