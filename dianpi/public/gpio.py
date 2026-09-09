#-*- coding: utf-8 -*-
from ctypes import *
import platform
from librockmong import *

# 初始化引脚工作模式
# SerialNumber: 设备序号
# Pin：引脚编号。0，P0. 1, P1...
# Mode：输入输出模式。0，输入。1，输出。2，开漏
# Pull：上拉下拉电阻。0，无。1，使能内部上拉。2，使能内部下拉
# 函数返回：0，正常；<0，异常
def IO_InitPin(SerialNumber, Pin, Mode, Pull):
    return librockmong.IO_InitPin(SerialNumber, Pin, Mode, Pull)
    
# 读取引脚状态
# SerialNumber: 设备序号
# Pin：引脚编号。0，P0. 1, P1...
# PinState：返回引脚状态。0，低电平。1，高电平
# 函数返回：0，正常；<0，异常
def IO_ReadPin(SerialNumber, Pin, PinState):
    return librockmong.IO_ReadPin(SerialNumber, Pin, PinState)

# 控制引脚输出状态
# SerialNumber: 设备序号
# Pin：引脚编号。0，P0. 1, P1...
# PinState：引脚状态。0，低电平。1，高电平
# 函数返回：0，正常；<0，异常
def IO_WritePin(SerialNumber, Pin, PinState):
    return librockmong.IO_WritePin(SerialNumber, Pin, PinState)
    
#读取输出引脚状态
#SerialNumber: 设备序号
#Pin：引脚编号。
#PinState：返回引脚状态。0，低电平。1，高电平
#函数返回：0，正常；<0，异常
def IO_ReadOutputPin(SerialNumber, Pin, PinState):
    return librockmong.IO_ReadOutputPin(SerialNumber, Pin, PinState)

class IO_InitMultiPin_TxStruct_t(Structure):  
	_fields_ = [
		("Pin", c_ubyte),	# 引脚编号
		("Mode", c_ubyte),	# 模式：0，输入；1，输出
		("Pull", c_ubyte)
	]

class IO_InitMultiPin_RxStruct_t(Structure):  
	_fields_ = [
		("Ret", c_ubyte),	# 返回
	]
    
# 批量初始化引脚
# SerialNumber: 设备序号
# TxStruct：输入数组。Pin 引脚编号；Mode：0输入 1输出 2开漏；Pull：0无 1上拉 2下拉
# RxStruct：输出数组。Ret：该引脚执行结果。0正常，<0异常
# Number：数组元素个数，即要操作的引脚数
# 函数返回：0，正常；<0，异常
def IO_InitMultiPin(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.IO_InitMultiPin(SerialNumber, TxStruct, RxStruct, Number)


class IO_ReadMultiPin_TxStruct_t(Structure):  
	_fields_ = [
		("Pin", c_ubyte),	# 引脚编号
	]

class IO_ReadMultiPin_RxStruct_t(Structure):  
	_fields_ = [
		("Ret", c_ubyte),		# 返回
		("PinState", c_ubyte),	# 引脚状态
	]
    
# 批量读取引脚状态
# SerialNumber: 设备序号
# TxStruct：输入数组。Pin 要读取的引脚编号
# RxStruct：输出数组。Ret 执行结果；PinState：0低电平，1高电平
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def IO_ReadMultiPin(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.IO_ReadMultiPin(SerialNumber, TxStruct, RxStruct, Number)


class IO_WriteMultiPin_TxStruct_t(Structure):  
	_fields_ = [
		("Pin", c_ubyte),		# 引脚编号
		("PinState", c_ubyte),	# 引脚状态
	]

class IO_WriteMultiPin_RxStruct_t(Structure):  
	_fields_ = [
		("Ret", c_ubyte),	#返回
	]
    
# 批量写引脚输出
# SerialNumber: 设备序号
# TxStruct：输入数组。Pin 引脚编号；PinState：0低电平，1高电平
# RxStruct：输出数组。Ret 执行结果
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def IO_WriteMultiPin(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.IO_WriteMultiPin(SerialNumber, TxStruct, RxStruct, Number)
	
	
class IO_ReadMultiOutputPin_TxStruct_t(Structure):  
	_fields_ = [
		("Pin", c_ubyte),	# 引脚编号
	]

class IO_ReadMultiOutputPin_RxStruct_t(Structure):  
	_fields_ = [
		("Ret", c_ubyte),		# 返回
		("PinState", c_ubyte),	# 引脚状态
	]
    
# 批量读取输出引脚当前设置的电平
# SerialNumber: 设备序号
# TxStruct：输入数组。Pin 引脚编号。0，P0. 1, P1...
# RxStruct：输出数组。Ret 执行结果；PinState：0低电平，1高电平
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def IO_ReadMultiOutputPin(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.IO_ReadMultiOutputPin(SerialNumber, TxStruct, RxStruct, Number)

# 同时写所有输出引脚（最多32路）
# SerialNumber: 设备序号
# PinState：每个 bit 对应一路。bit0=P0，bit1=P1...。0低电平，1高电平
# 函数返回：0，正常；<0，异常
def IO_WritePin_Bit(SerialNumber, PinState):
    return librockmong.IO_WritePin_Bit(SerialNumber, PinState)

# 同时读取所有输出引脚当前设置（最多32路）
# SerialNumber: 设备序号
# PinState：c_int，返回位图，含义同 IO_WritePin_Bit
# 函数返回：0，正常；<0，异常
def IO_ReadOutputPin_Bit(SerialNumber, PinState):
    return librockmong.IO_ReadOutputPin_Bit(SerialNumber, PinState)

# 同时读取所有输入引脚状态（最多32路）
# SerialNumber: 设备序号
# PinState：c_int，返回位图。bit0=P0，bit1=P1...。0低电平，1高电平
# 函数返回：0，正常；<0，异常
def IO_ReadPin_Bit(SerialNumber, PinState):
    return librockmong.IO_ReadPin_Bit(SerialNumber, PinState)
	
