#-*- coding: utf-8 -*-
from ctypes import *
import os
import sys
public_path = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + "/../public")
sys.path.append(public_path)
from librockmong import *
from device import *
from uart import *

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

    # 初始化 UART0：115200 8N1，无硬件流控（设备对外串口，不是电脑侧 COM）
    ret = UART_Init(sn, 0, 115200, 8, 1, 0, 0)
    if (0 > ret):
        print("error: %d"%ret)

    ret = UART_Clear(sn, 0)
    if (0 > ret):
        print("error: %d"%ret)

    # 发送 Hello
    tx = b"Hello"
    TxBuffer = (c_ubyte * len(tx)).from_buffer_copy(tx)
    ret = UART_Write(sn, 0, TxBuffer, len(tx))
    if (0 > ret):
        print("error: %d"%ret)
    else:
        print("UART Write %d bytes"%len(tx))

    # 读取，超时 1s。若 TX/RX 未短接，ActualLength 可能为 0
    RxBuffer = (c_ubyte * 64)()
    ActualLength = c_int(0)
    ret = UART_Read(sn, 0, RxBuffer, byref(ActualLength), 64, 1000)
    if (0 > ret):
        print("error: %d"%ret)
    else:
        print("UART Read %d bytes:"%ActualLength.value, " ".join("0x%02X"%RxBuffer[i] for i in range(ActualLength.value)))

    # 关闭设备
    ret = Device_Close(sn)
    if (0 > ret):
        print("error: %d"%ret)
