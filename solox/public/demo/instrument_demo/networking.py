import json
import os
import queue
import sys


sys.path.append(os.getcwd())
import time
from _ctypes import Structure
from ctypes import c_byte, c_uint16, c_uint32
from ios_device.servers.Instrument import  InstrumentServer
from ios_device.util import logging

log = logging.getLogger(__name__)


def networking(rpc, result_queue):
    headers = {
        0: ['InterfaceIndex', "Name"],
        1: ['LocalAddress', 'RemoteAddress', 'InterfaceIndex', 'Pid', 'RecvBufferSize', 'RecvBufferUsed',
            'SerialNumber', 'Kind'],
        2: ['RxPackets', 'RxBytes', 'TxPackets', 'TxBytes', 'RxDups', 'RxOOO', 'TxRetx', 'MinRTT', 'AvgRTT',
            'ConnectionSerial']
    }
    msg_type = {
        0: "interface-detection",
        1: "connection-detected",
        2: "connection-update",
    }

    def on_callback_message(res):
        from socket import inet_ntoa, htons, inet_ntop, AF_INET6
        class SockAddr4(Structure):
            _fields_ = [
                ('len', c_byte),
                ('family', c_byte),
                ('port', c_uint16),
                ('addr', c_byte * 4),
                ('zero', c_byte * 8)
            ]

            def __str__(self):
                return f"{inet_ntoa(self.addr)}:{htons(self.port)}"

        class SockAddr6(Structure):
            _fields_ = [
                ('len', c_byte),
                ('family', c_byte),
                ('port', c_uint16),
                ('flowinfo', c_uint32),
                ('addr', c_byte * 16),
                ('scopeid', c_uint32)
            ]

            def __str__(self):
                return f"[{inet_ntop(AF_INET6, self.addr)}]:{htons(self.port)}"

        data = res.selector
        # if data[0] == 1:
        #     if len(data[1][0]) == 16:
        #         data[1][0] = str(SockAddr4.from_buffer_copy(data[1][0]))
        #         data[1][1] = str(SockAddr4.from_buffer_copy(data[1][1]))
        #     elif len(data[1][0]) == 28:
        #         data[1][0] = str(SockAddr6.from_buffer_copy(data[1][0]))
        #         data[1][1] = str(SockAddr6.from_buffer_copy(data[1][1]))

        # print(msg_type[data[0]] + json.dumps(dict(zip(headers[data[0]], data[1]))))

        if data[0] == 2:
            rx_packets, rx_bytes, tx_packets, tx_bytes = data[1][:4]
        # for _type, values in it:
        #     print(values)
        #     (_type, values) = data
        #     if _type == 2:
        #         rx_packets, rx_bytes, tx_packets, tx_bytes = values[:4]
        #     yield {
        #         "rx.packets": rx_packets,
        #         "rx.bytes": rx_bytes,
        #         "tx.packets": tx_packets,
        #         "tx.bytes": tx_bytes,
        #     }
            result_queue.put({
                        "rx.packets": rx_packets,
                        "rx.bytes": rx_bytes,
                        "tx.packets": tx_packets,
                        "tx.bytes": tx_bytes,
                    })
        #         print("[data]", res.selector)

    rpc.register_channel_callback("com.apple.instruments.server.services.networking", on_callback_message)
    var = rpc.call("com.apple.instruments.server.services.networking", "replayLastRecordedSession").selector
    log.debug(f"replay {var}")
    var = rpc.call("com.apple.instruments.server.services.networking", "startMonitoring").selector
    log.debug(f"start {var}")
    time.sleep(0.5)
    try:
        result = result_queue.get(timeout=5)  # 等待 5 秒
    except queue.Empty:
        print("Timeout: No message received from on_sysmontap_message.")
        result = None
    var = rpc.call("com.apple.instruments.server.services.networking", "stopMonitoring").selector
    log.debug(f"stopMonitoring {var}")
    rpc.stop()
    return result


if __name__ == '__main__':
    rpc = InstrumentServer().init()
    networking(rpc)
    rpc.stop()
