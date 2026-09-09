#-*- coding: utf-8 -*-
from ctypes import *
from librockmong import *

# 初始化PWM
# SerialNumber: 设备序号
# Channel: 通道编号。0，PWM0. 1, PWM1 ...
# Frequency：频率/Hz。最小1Hz, 最大1MHz
# Duty: 占空比。千分比0~1000
# Phase: 波形相位。千分比0~1000
# Polarity: 波形极性。范围0~1。
# 函数返回：0，正常；<0，异常
def PWM_Init(SerialNumber, Channel, Frequency, Duty, Phase, Polarity):
    return librockmong.PWM_Init(SerialNumber, Channel, Frequency, Duty, Phase, Polarity)

# PWM开始输出
# SerialNumber: 设备序号
# Channel: 通道编号。0，PWM0. 1, PWM1 ...
# RunTimeUs: 输出波形的时间，单位为微秒。启动后经过该时间停止输出；为 0 则一直输出直到 PWM_Stop。可用该参数控制脉冲个数
# 函数返回：0，正常；<0，异常
def PWM_Start(SerialNumber, Channel, RunTimeUs):
    return librockmong.PWM_Start(SerialNumber, Channel, RunTimeUs)

# PWM停止输出
# Channel: 通道编号。0，PWM0. 1, PWM1 ...
# 函数返回：0，正常；<0，异常
def PWM_Stop(SerialNumber, Channel):
    return librockmong.PWM_Stop(SerialNumber, Channel)

# PWM占空比动态调节。可以在PWM启动之后调用
# Channel: 通道编号。0，PWM0. 1, PWM1 ...
# Duty: 占空比。千分比0~1000
# 函数返回：0，正常；<0，异常
def PWM_SetDuty(SerialNumber, Channel, Duty):
    return librockmong.PWM_SetDuty(SerialNumber, Channel, Duty)

# PWM频率动态调节。可以在PWM启动之后调用
# Channel: 通道编号。0，PWM0. 1, PWM1 ...
# Frequency：频率/Hz。最小1Hz, 最大1MHz
# 函数返回：0，正常；<0，异常
def PWM_SetFrequency(SerialNumber, Channel, Frequency):
    return librockmong.PWM_SetFrequency(SerialNumber, Channel, Frequency)
	
# PWM相位动态调节。可以在PWM启动之后调用
# Channel: 通道编号。0，PWM0. 1, PWM1 ...
# Phase: 波形相位。千分比0~1000
# 函数返回：0，正常；<0，异常
def PWM_SetPhase(SerialNumber, Channel, Phase):
    return librockmong.PWM_SetPhase(SerialNumber, Channel, Phase)

# 读取剩余脉冲个数。配合 PWM_Start 的 WaveCount/RunTimeUs 使用
# Channel: 通道编号。0，PWM0. 1, PWM1 ...
# Number：返回还剩多少个脉冲未输出
# 函数返回：0，正常；<0，异常
def PWM_GetNumberOfRemainingPulses(SerialNumber, Channel, Number):
    return librockmong.PWM_GetNumberOfRemainingPulses(SerialNumber, Channel, Number)

class PWM_Start_withList_TxStruct_t(Structure):  
	_fields_ = [
		("Channel", c_ubyte),		# 通道编号
		("Frequency", c_uint32),		# 频率
		("Duty", c_ushort),			# 占空比
		("WaveCount", c_ushort),	# 波的个数
	]

class PWM_Start_withList_RxStruct_t(Structure):  
	_fields_ = [
		("Ret", c_ubyte),	#返回
	]

# 按列表输出PWM。可同时指定频率、占空比和脉冲个数
# SerialNumber: 设备序号
# TxStruct：输入数组。Channel 通道；Frequency 频率Hz；Duty 占空比千分比；WaveCount 要输出的脉冲个数
# RxStruct：输出数组。Ret 执行结果
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def PWM_Start_withList(SerialNumber, TxStruct, RxStruct, Number):
	return librockmong.PWM_Start_withList(SerialNumber, TxStruct, RxStruct, Number)