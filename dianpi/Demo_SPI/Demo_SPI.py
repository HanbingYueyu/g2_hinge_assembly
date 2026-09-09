#-*- coding: utf-8 -*-
from ctypes import *
import os
import sys
public_path = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + "/../public")
sys.path.append(public_path)
from librockmong import *
from device import *
from spi import *

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
    spi_channel = 0
    ret = Device_Open(sn)
    if (0 > ret):
        print("error: %d"%ret)
        exit()

    # 初始化SPI0：主机，1MHz，CPHA=0，CPOL=0，片选低有效
    ret = SPI_Init(sn, spi_channel, 0, 1000000, 0, 0, 0)
    if (0 > ret):
        print("SPI Init error: %d"%ret)
        Device_Close(sn)
        exit()
    print("SPI initialized")

    writeBuffer = (c_int * 3)(0x01, 0x02, 0x03)
    ret = SPI_WriteBytes(sn, spi_channel, writeBuffer, 3)
    if (0 > ret):
        print("SPI Write error: %d"%ret)
    else:
        print("SPI Write %d bytes"%3)

    readBuffer = (c_int * 4)()
    ret = SPI_ReadBytes(sn, spi_channel, readBuffer, 4)
    if (0 > ret):
        print("SPI Read error: %d"%ret)
    else:
        print("SPI Read:", " ".join("0x%02X"%readBuffer[i] for i in range(4)))

    writeRead = (c_int * 2)(0xAA, 0xBB)
    readRead = (c_int * 2)()
    ret = SPI_WriteReadBytes(sn, spi_channel, writeRead, readRead, 2, 2)
    if (0 > ret):
        print("SPI WriteRead error: %d"%ret)
    else:
        print("SPI WriteRead:", " ".join("0x%02X"%readRead[i] for i in range(2)))

    ret = Device_Close(sn)
    if (0 > ret):
        print("error: %d"%ret)
