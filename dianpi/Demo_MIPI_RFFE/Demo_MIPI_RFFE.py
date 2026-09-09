#-*- coding: utf-8 -*-
from ctypes import *
import os
import sys
public_path = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + "/../public")
sys.path.append(public_path)
from librockmong import *
from device import *
from gpio import *
from mipi_rffe import *

if __name__ == '__main__':
    SerialNumbers = (c_int * 20)()
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

    sn = SerialNumbers[0]
    rffe_channel = 7
    rffe_slave_address = 0x08
    rffe_register_address = 0x1F
    ret = Device_Open(sn)
    if (0 > ret):
        print("error: %d"%ret)
        exit()

    # P0 接模块 VIO；SCLK=P14，SDATA=P15，对应通道 7
    ret = IO_InitPin(sn, 0, 1, 0)
    if (0 > ret):
        print("error: %d"%ret)
    ret = IO_WritePin(sn, 0, 0)
    if (0 > ret):
        print("error: %d"%ret)

    ret = RFFE_Init(sn, rffe_channel, 1000000)
    if (0 > ret):
        print("RFFE Init error: %d"%ret)

    ret = IO_WritePin(sn, 0, 1)
    if (0 > ret):
        print("error: %d"%ret)

    value = (c_int)()
    ret = RFFE_Reg_Read(sn, rffe_channel, rffe_slave_address, rffe_register_address, byref(value))
    if (0 > ret):
        print("RFFE Read error: %d"%ret)
    else:
        print("Read value: 0x%02X"%value.value)

    ret = Device_Close(sn)
    if (0 > ret):
        print("error: %d"%ret)
