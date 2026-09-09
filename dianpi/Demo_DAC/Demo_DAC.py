#-*- coding: utf-8 -*-
from ctypes import *
import os
import sys
public_path = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + "/../public")
sys.path.append(public_path)
from librockmong import *
from device import *
from dac import *

if __name__ == '__main__':
    SerialNumbers = (c_int * 20)()
    sn = 0
    # Scan device
    # 若感觉 Device_Scan 太慢，可按需改用 UsbDevice_Scan、NetDevice_Scan、CH343Dev_Scan
    ret = Device_Scan(byref(SerialNumbers))
    if (0 > ret):
        print("Error: %d"%ret)
        exit()
    elif(ret == 0):
        print("No device!")
        exit()
    else:
        for i in range(ret):
            print("Dev%d SN: %d"%(i, SerialNumbers[i]))
    
    sn = SerialNumbers[0]   # 选择设备0

    # 打开设备
    ret = Device_Open(sn)
    if (0 > ret):
        print("error: %d"%ret)
        exit()

    # 生成 32 点锯齿波，16 位满幅（库内转成 16 位）
    Size = 32
    Buffer = (c_int * Size)()
    for i in range(Size):
        Buffer[i] = i * 65535 // (Size - 1)

    # 初始化DAC0，下载波形。相邻两点间隔 10us，不会立即输出
    ret = DAC_Init(sn, 0, 10000, Buffer, Size)
    if (0 > ret):
        print("error: %d"%ret)

    # 开始DAC输出
    ret = DAC_Start(sn, 0)
    if (0 > ret):
        print("error: %d"%ret)

    # 停止DAC输出
    ret = DAC_Stop(sn, 0)
    if (0 > ret):
        print("error: %d"%ret)

    # 关闭设备
    ret = Device_Close(sn)
    if (0 > ret):
        print("error: %d"%ret)
