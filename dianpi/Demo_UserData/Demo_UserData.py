#-*- coding: utf-8 -*-
from ctypes import *
import os
import sys
public_path = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + "/../public")
sys.path.append(public_path)
from librockmong import *
from device import *
from user_data import *

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
    addr = 100
    size = 256
    ret = Device_Open(sn)
    if (0 > ret):
        print("error: %d"%ret)
        exit()

    # 写入前先擦除用户数据区
    ret = UserData_Erase(sn)
    if (0 > ret):
        print("Erase error: %d"%ret)

    writeDat = (c_ubyte * size)(*range(size))
    ret = UserData_Write(sn, addr, size, writeDat)
    if (0 > ret):
        print("Write error: %d"%ret)

    readDat = (c_ubyte * size)()
    ret = UserData_Read(sn, addr, size, readDat)
    if (0 > ret):
        print("Read error: %d"%ret)

    mismatch = 0
    for i in range(size):
        if writeDat[i] != readDat[i]:
            print("Error addr: %d  %d != %d"%(addr + i, writeDat[i], readDat[i]))
            mismatch += 1
            if mismatch >= 8:
                break
    if mismatch == 0:
        print("UserData read/write match, %d bytes"%size)

    ret = Device_Close(sn)
    if (0 > ret):
        print("error: %d"%ret)
