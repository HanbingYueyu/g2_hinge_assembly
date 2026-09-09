#-*- coding: utf-8 -*-
from ctypes import *
from librockmong import *

def _enc(s):
    if isinstance(s, str):
        return s.encode("utf-8")
    return s

# 设置是否启用 MQTT。Enable：0，关闭。1，使能
# SerialNumber: 设备序号。函数返回：0，正常；<0，异常
def MQTT_Set_Enable(SerialNumber, Enable):
    return librockmong.MQTT_Set_Enable(SerialNumber, Enable)

# 读取是否启用 MQTT。Enable：c_int，返回 0 关闭，1 使能
def MQTT_Get_Enable(SerialNumber, Enable):
    return librockmong.MQTT_Get_Enable(SerialNumber, Enable)

# 设置/读取 MQTT 服务器域名。缓冲区建议不少于 64 字节
def MQTT_Set_ServerUrl(SerialNumber, ServerUrl):
    return librockmong.MQTT_Set_ServerUrl(SerialNumber, _enc(ServerUrl))

def MQTT_Get_ServerUrl(SerialNumber, ServerUrl):
    return librockmong.MQTT_Get_ServerUrl(SerialNumber, ServerUrl)

# 设置/读取 MQTT 服务器 IP。IPv4 字符串，建议不少于 16 字节
def MQTT_Set_ServerIP(SerialNumber, ServerIP):
    return librockmong.MQTT_Set_ServerIP(SerialNumber, _enc(ServerIP))

def MQTT_Get_ServerIP(SerialNumber, ServerIP):
    return librockmong.MQTT_Get_ServerIP(SerialNumber, ServerIP)

# 设置/读取 MQTT 服务器端口
def MQTT_Set_ServerPort(SerialNumber, ServerPort):
    return librockmong.MQTT_Set_ServerPort(SerialNumber, ServerPort)

def MQTT_Get_ServerPort(SerialNumber, ServerPort):
    return librockmong.MQTT_Get_ServerPort(SerialNumber, ServerPort)

# 设置/读取 ClientID。缓冲区建议不少于 64 字节
def MQTT_Set_ClientID(SerialNumber, ClientID):
    return librockmong.MQTT_Set_ClientID(SerialNumber, _enc(ClientID))

def MQTT_Get_ClientID(SerialNumber, ClientID):
    return librockmong.MQTT_Get_ClientID(SerialNumber, ClientID)

# 设置/读取用户名。缓冲区建议不少于 64 字节
def MQTT_Set_Username(SerialNumber, Username):
    return librockmong.MQTT_Set_Username(SerialNumber, _enc(Username))

def MQTT_Get_Username(SerialNumber, Username):
    return librockmong.MQTT_Get_Username(SerialNumber, Username)

# 设置/读取密码。缓冲区建议不少于 64 字节
def MQTT_Set_Passwd(SerialNumber, Passwd):
    return librockmong.MQTT_Set_Passwd(SerialNumber, _enc(Passwd))

def MQTT_Get_Passwd(SerialNumber, Passwd):
    return librockmong.MQTT_Get_Passwd(SerialNumber, Passwd)

# 设置/读取发布主题。缓冲区建议不少于 64 字节
def MQTT_Set_PubTopic(SerialNumber, PubTopic):
    return librockmong.MQTT_Set_PubTopic(SerialNumber, _enc(PubTopic))

def MQTT_Get_PubTopic(SerialNumber, PubTopic):
    return librockmong.MQTT_Get_PubTopic(SerialNumber, PubTopic)

# 设置/读取订阅主题。缓冲区建议不少于 64 字节
def MQTT_Set_SubTopic(SerialNumber, SubTopic):
    return librockmong.MQTT_Set_SubTopic(SerialNumber, _enc(SubTopic))

def MQTT_Get_SubTopic(SerialNumber, SubTopic):
    return librockmong.MQTT_Get_SubTopic(SerialNumber, SubTopic)

# 设置/读取发布 QoS。取值 0、1、2
def MQTT_Set_PubQoS(SerialNumber, PubQoS):
    return librockmong.MQTT_Set_PubQoS(SerialNumber, PubQoS)

def MQTT_Get_PubQoS(SerialNumber, PubQoS):
    return librockmong.MQTT_Get_PubQoS(SerialNumber, PubQoS)

# 设置/读取订阅 QoS。取值 0、1、2
def MQTT_Set_SubQoS(SerialNumber, SubQoS):
    return librockmong.MQTT_Set_SubQoS(SerialNumber, SubQoS)

def MQTT_Get_SubQoS(SerialNumber, SubQoS):
    return librockmong.MQTT_Get_SubQoS(SerialNumber, SubQoS)
