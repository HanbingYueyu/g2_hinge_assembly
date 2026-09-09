#-*- coding: utf-8 -*-
from ctypes import *
from librockmong import *

# 初始化ADC
# SerialNumber: 设备序号
# Channel：通道编号。0，ADC0. 1, ADC1...
# SampleRateHz：采样率。一般设置0
# 函数返回：0，正常；<0，异常
def ADC_Init(SerialNumber, Channel, SampleRateHz):
    return librockmong.ADC_Init(SerialNumber, Channel, SampleRateHz)

# ADC读取
# SerialNumber: 设备序号
# Channel：通道编号。0，ADC0. 1, ADC1...
# Value：AD值
# 函数返回：0，正常；<0，异常
def ADC_Read(SerialNumber, Channel, Value):
    return librockmong.ADC_Read(SerialNumber, Channel, Value)

# 连续读取ADC缓冲区
# SerialNumber: 设备序号
# Channel：通道编号。0，ADC0. 1, ADC1...
# Buffer：输出数组，存放 AD 值。调用前请分配 BufferSize 个 int
# BufferSize：最多读取的点数
# Length：c_int，返回实际读到的点数
# 函数返回：0，正常；<0，异常
def ADC_ReadBuffer(SerialNumber, Channel, Buffer, BufferSize, Length):
    return librockmong.ADC_ReadBuffer(SerialNumber, Channel, Buffer, BufferSize, Length)

class ADC_Init_TxStruct_t(Structure):  
	_fields_ = [
		("Channel", c_ubyte),		# 通道编号
		("SampleRateHz", c_uint32),	# 采样率
	]

class ADC_Init_RxStruct_t(Structure):  
	_fields_ = [
		("Ret", c_ubyte),	#返回
	]
        
# 批量初始化ADC
# SerialNumber: 设备序号
# TxStruct：输入数组。Channel：通道编号；SampleRateHz：采样率，一般填 0
# RxStruct：输出数组。Ret：该通道执行结果
# Number：数组元素个数，即通道个数
# 函数返回：0，正常；<0，异常
def ADC_InitMulti(SerialNumber, TxStruct, RxStruct, Number):
	return librockmong.ADC_InitMulti(SerialNumber, TxStruct, RxStruct, Number)

class ADC_Read_TxStruct_t(Structure):  
	_fields_ = [
		("Channel", c_ubyte),		# 通道编号
	]

class ADC_Read_RxStruct_t(Structure):  
	_fields_ = [
		("Ret", c_ubyte),	    #返回
		("Value", c_uint16),	#ADC数值
	]
	
# 批量读取ADC
# SerialNumber: 设备序号
# TxStruct：输入数组。Channel：通道编号
# RxStruct：输出数组。Ret 执行结果；Value：该通道 AD 值
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def ADC_ReadMulti(SerialNumber, TxStruct, RxStruct, Number):
	return librockmong.ADC_ReadMulti(SerialNumber, TxStruct, RxStruct, Number)
