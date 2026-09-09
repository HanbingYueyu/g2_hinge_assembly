#-*- coding: utf-8 -*-
from ctypes import *
from librockmong import *

# 设置端口电平（电源电压）
# SerialNumber: 设备序号
# Millivolt：电压，单位 mV
# 函数返回：0，正常；<0，异常
def PwrOn_SetPortLevel(SerialNumber, Millivolt):
    return librockmong.PwrOn_SetPortLevel(SerialNumber, Millivolt)

# 读取端口电平（电源电压）
# SerialNumber: 设备序号
# Millivolt：返回电压，单位 mV
# 函数返回：0，正常；<0，异常
def PwrOn_GetPortLevel(SerialNumber, Millivolt):
    return librockmong.PwrOn_GetPortLevel(SerialNumber, Millivolt)

# 恢复 IO 上电默认为出厂设置
# SerialNumber: 设备序号
# 函数返回：0，正常；<0，异常
def PwrOn_DefaultIO(SerialNumber):
    return librockmong.PwrOn_DefaultIO(SerialNumber)

# 恢复 PWM 上电默认为出厂设置
# SerialNumber: 设备序号
# 函数返回：0，正常；<0，异常
def PwrOn_DefaultPWM(SerialNumber):
    return librockmong.PwrOn_DefaultPWM(SerialNumber)

# 设置上电时 PWM 是否自动输出
# SerialNumber: 设备序号
# Channel：通道编号。0，PWM0. 1, PWM1...
# Enable：0，上电不输出。1，上电自动输出
# Frequency：频率 Hz
# Duty：占空比。千分比 0~1000
# Phase：相位。千分比 0~1000
# Polarity：波形极性。0 或 1
# 函数返回：0，正常；<0，异常
def PwrOn_PWM(SerialNumber, Channel, Enable, Frequency, Duty, Phase, Polarity):
    return librockmong.PwrOn_PWM(SerialNumber, Channel, Enable, Frequency, Duty, Phase, Polarity)

class PwrOn_IO_TxStruct_t(Structure):
    _fields_ = [
        ("Pin", c_ubyte),		# 引脚编号。0，P0. 1, P1...
        ("Mode", c_ubyte),		# 0，输入。1，输出。2，开漏
        ("PuPd", c_ubyte),		# 0，无。1，上拉。2，下拉
        ("PinState", c_ubyte),	# 上电后的电平。0，低。1，高
    ]

class PwrOn_IO_RxStruct_t(Structure):
    _fields_ = [
        ("Ret", c_ubyte),		# 返回。0，正常；<0，异常
    ]

# 批量设置上电时 IO 状态
# SerialNumber: 设备序号
# TxStruct：每个引脚的模式、上下拉、电平
# RxStruct：每个引脚的执行结果
# Number：引脚个数
# 函数返回：0，正常；<0，异常
def PwrOn_IO_Multi(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.PwrOn_IO_Multi(SerialNumber, TxStruct, RxStruct, Number)

class PwrOn_SetIO_TxStruct_t(Structure):
    _fields_ = [
        ("Pin", c_ubyte),
        ("Mode", c_ubyte),
        ("PuPd", c_ubyte),
        ("PinState", c_ubyte),
    ]

class PwrOn_SetIO_RxStruct_t(Structure):
    _fields_ = [
        ("Ret", c_byte),
    ]

# 批量写入上电 IO 配置（与 PwrOn_IO_Multi 字段含义相同）
# SerialNumber: 设备序号
# TxStruct：输入数组。Pin 引脚；Mode：0输入 1输出 2开漏；PuPd：0无 1上拉 2下拉；PinState：上电后电平 0低 1高
# RxStruct：输出数组。Ret 执行结果
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def PwrOn_SetIO_Multi(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.PwrOn_SetIO_Multi(SerialNumber, TxStruct, RxStruct, Number)

class PwrOn_GetIO_TxStruct_t(Structure):
    _fields_ = [
        ("Pin", c_ubyte),		# 要查询的引脚
    ]

class PwrOn_GetIO_RxStruct_t(Structure):
    _fields_ = [
        ("Ret", c_byte),
        ("Pin", c_ubyte),
        ("Mode", c_ubyte),
        ("PuPd", c_ubyte),
        ("PinState", c_ubyte),
    ]

# 批量读取上电 IO 配置
# SerialNumber: 设备序号
# TxStruct：输入数组。Pin 要查询的引脚
# RxStruct：输出数组。Ret 执行结果；Pin/Mode/PuPd/PinState 为该引脚当前上电配置
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def PwrOn_GetIO_Multi(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.PwrOn_GetIO_Multi(SerialNumber, TxStruct, RxStruct, Number)

class PwrOn_PWM_TxStruct_t(Structure):
    _fields_ = [
        ("Channel", c_ubyte),		# 通道编号
        ("Enable", c_ubyte),		# 0，关闭。1，使能
        ("Frequency", c_uint32),	# 频率 Hz
        ("Duty", c_uint16),			# 占空比。千分比
        ("Phase", c_uint16),		# 相位。千分比
        ("Polarity", c_ubyte),		# 波形极性
    ]

class PwrOn_PWM_RxStruct_t(Structure):
    _fields_ = [
        ("Ret", c_byte),
    ]

# 批量设置上电 PWM
# SerialNumber: 设备序号
# TxStruct：输入数组。Channel 通道；Enable：0上电不输出 1上电自动输出；Frequency 频率Hz；Duty/Phase 千分比0~1000；Polarity 0或1
# RxStruct：输出数组。Ret 执行结果
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def PwrOn_PWM_Multi(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.PwrOn_PWM_Multi(SerialNumber, TxStruct, RxStruct, Number)

class PwrOn_GetPWM_TxStruct_t(Structure):
    _fields_ = [
        ("Channel", c_ubyte),
    ]

class PwrOn_GetPWM_RxStruct_t(Structure):
    _fields_ = [
        ("Ret", c_byte),
        ("Channel", c_ubyte),
        ("Enable", c_ubyte),
        ("Frequency", c_uint32),
        ("Duty", c_uint16),
        ("Phase", c_uint16),
        ("Polarity", c_ubyte),
    ]

# 批量读取上电 PWM 配置
# SerialNumber: 设备序号
# TxStruct：输入数组。Channel 要查询的通道
# RxStruct：输出数组。Ret 执行结果；其余字段为该通道当前上电 PWM 配置
# Number：数组元素个数
# 函数返回：0，正常；<0，异常
def PwrOn_GetPWM_Multi(SerialNumber, TxStruct, RxStruct, Number):
    return librockmong.PwrOn_GetPWM_Multi(SerialNumber, TxStruct, RxStruct, Number)
