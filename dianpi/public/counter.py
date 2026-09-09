#-*- coding: utf-8 -*-
from ctypes import *
import platform
from librockmong import *

# 初始化计数器
# SerialNumber: 设备序号
# Channel：通道编号。0，计数器0. 1, 计数器1...
# Mode：计数边沿。0，上升沿。1，下降沿。2，双边沿
# Dirction：计数方向。0，加。1，减
# Pull：上拉下拉电阻。0，无。1，内部上拉。2，内部下拉
# 函数返回：0，正常；<0，异常
def Counter_Init(SerialNumber, Channel, Mode, Dirction, Pull):
	return librockmong.Counter_Init(SerialNumber, Channel, Mode, Dirction, Pull)

# 开始计数
# SerialNumber: 设备序号
# Channel：通道编号
# 函数返回：0，正常；<0，异常
def Counter_Start(SerialNumber, Channel):
	return librockmong.Counter_Start(SerialNumber, Channel)

# 停止计数
# SerialNumber: 设备序号
# Channel：通道编号
# 函数返回：0，正常；<0，异常
def Counter_Stop(SerialNumber, Channel):
	return librockmong.Counter_Stop(SerialNumber, Channel)

# 读取计数值
# SerialNumber: 设备序号
# Channel：通道编号
# Value：返回当前计数值
# 函数返回：0，正常；<0，异常
def Counter_Read(SerialNumber, Channel, Value):
	return librockmong.Counter_Read(SerialNumber, Channel, Value)

# 写入计数值
# SerialNumber: 设备序号
# Channel：通道编号
# Value：要写入的计数值
# 函数返回：0，正常；<0，异常
def Counter_Write(SerialNumber, Channel, Value):
	return librockmong.Counter_Write(SerialNumber, Channel, Value)

class Counter_InitMulti_TxStruct_t(Structure):  
	_fields_ = [
		("Channel", c_ubyte),	# 通道编号
		("Mode", c_ubyte),
		("Direction", c_ubyte),
		("Pull", c_ubyte)
	]

class Counter_InitMulti_RxStruct_t(Structure):  
	_fields_ = [
		("Ret", c_ubyte),	# 返回
	]

# 批量初始化计数器
# SerialNumber: 设备序号
# TxStruct：输入数组。Channel 通道；Mode：0上升沿 1下降沿 2双边沿；Direction：0加 1减；Pull：0无 1上拉 2下拉
# RxStruct：输出数组。Ret：该通道执行结果
# Number：数组元素个数，即通道个数
# 函数返回：0，正常；<0，异常
def Counter_InitMulti(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.Counter_InitMulti(SerialNumber, TxStruct, RxStruct, Number)

class Counter_StartMulti_TxStruct_t(Structure):  
	_fields_ = [
		("Channel", c_ubyte),	# 通道编号
	]

class Counter_StartMulti_RxStruct_t(Structure):  
	_fields_ = [
		("Ret", c_ubyte),	# 返回
	]

# 批量开始计数
# SerialNumber: 设备序号
# TxStruct：输入数组。Channel 通道
# RxStruct：输出数组。Ret 执行结果
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def Counter_StartMulti(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.Counter_StartMulti(SerialNumber, TxStruct, RxStruct, Number)

class Counter_StopMulti_TxStruct_t(Structure):  
	_fields_ = [
		("Channel", c_ubyte),	# 通道编号
	]

class Counter_StopMulti_RxStruct_t(Structure):  
	_fields_ = [
		("Ret", c_ubyte),	# 返回
	]

# 批量停止计数
# SerialNumber: 设备序号
# TxStruct：输入数组。Channel 通道
# RxStruct：输出数组。Ret 执行结果
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def Counter_StopMulti(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.Counter_StopMulti(SerialNumber, TxStruct, RxStruct, Number)

class Counter_ReadMulti_TxStruct_t(Structure):  
	_fields_ = [
		("Channel", c_ubyte),	# 通道编号
	]

class Counter_ReadMulti_RxStruct_t(Structure):  
	_fields_ = [
		("Ret", c_ubyte),		# 返回
		("Value", c_uint32),	# 
	]

# 批量读取计数值
# SerialNumber: 设备序号
# TxStruct：输入数组。Channel 通道
# RxStruct：输出数组。Ret 执行结果；Value：该通道当前计数值
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def Counter_ReadMulti(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.Counter_ReadMulti(SerialNumber, TxStruct, RxStruct, Number)

class Counter_WriteMulti_TxStruct_t(Structure):  
	_fields_ = [
		("Channel", c_ubyte),	# 通道编号
		("Value", c_uint32),	# 
	]

class Counter_WriteMulti_RxStruct_t(Structure):  
	_fields_ = [
		("Ret", c_ubyte),		# 返回
	]

# 批量写入计数值
# SerialNumber: 设备序号
# TxStruct：输入数组。Channel 通道；Value 要写入的计数值
# RxStruct：输出数组。Ret 执行结果
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def Counter_WriteMulti(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.Counter_WriteMulti(SerialNumber, TxStruct, RxStruct, Number)