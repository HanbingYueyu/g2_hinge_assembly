from ctypes import *
import platform
import os
import shutil

current_path = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))

#根据系统自动导入对应的库文件，若没能识别到正确的系统，可以修改下面的源码
if(platform.system()=="Windows"):
    if "64bit" in platform.architecture():
        librockmong = windll.LoadLibrary(current_path+"/libs/windows/x86_64/librockmong.dll" )
    else:
        librockmong = windll.LoadLibrary(current_path+"/libs/windows/x86/librockmong.dll" )
elif(platform.system()=="Darwin"):
    if platform.machine() in ("arm64", "aarch64"):
        mac_dir = current_path+"/libs/mac_os/arm64"
    else:
        mac_dir = current_path+"/libs/mac_os/x86_64"
    librockmong = cdll.LoadLibrary(mac_dir+"/librockmong.dylib")
elif(platform.system()=="Linux"):
    if 'armv7' in platform.machine():
        librockmong = cdll.LoadLibrary(current_path+"/libs/linux/armv7/librockmong.so" )
    elif 'mips64' in platform.machine():
        librockmong = cdll.LoadLibrary(current_path+"/libs/linux/mips64/librockmong.so" )
    elif 'aarch64' in platform.machine():
        librockmong = cdll.LoadLibrary(current_path+"/libs/linux/aarch64/librockmong.so" )
    elif 'arm64' in platform.machine():
        librockmong = cdll.LoadLibrary(current_path+"/libs/linux/arm64/librockmong.so" )
    else:
        if "64bit" in platform.architecture():
            librockmong = cdll.LoadLibrary(current_path+"/libs/linux/x86_64/librockmong.so" )
        else:
            librockmong = cdll.LoadLibrary(current_path+"/libs/linux/x86/librockmong.so" )
else:
    print("unsupported system")
    exit()


