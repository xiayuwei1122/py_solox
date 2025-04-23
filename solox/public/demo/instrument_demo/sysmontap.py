"""
获取系统相关信息，类似 Android TOP，包含进程信息，需要 iOS > 11
"""
import os
import sys
import time
from typing import Any, Iterator, List, Optional, Tuple, Union

sys.path.append(os.getcwd())

import threading
import queue

def sysmontap(rpc, result_queue):
    def dropped_message(res):
        print("[DROP]", res.selector, res.raw.channel_code)

    def on_sysmontap_message(res):
        if isinstance(res.selector, list):
            result_queue.put(res.selector)

    rpc.register_undefined_callback(dropped_message)
    config = {
        "bm": 0,
        "cpuUsage": True,
        "procAttrs": [
            "memVirtualSize", "cpuUsage", "ctxSwitch", "intWakeups",
            "physFootprint", "memResidentSize", "memAnon", "pid"
        ],
        "sampleInterval": 1000000000,  # 1e9 ns == 1s
        "sysAttrs": [
            "vmExtPageCount", "vmFreeCount", "vmPurgeableCount",
            "vmSpeculativeCount", "physMemSize"
        ],
        "ur": 1000
    }
    rpc.call("com.apple.instruments.server.services.sysmontap", "setConfig:", config)
    rpc.register_channel_callback("com.apple.instruments.server.services.sysmontap", on_sysmontap_message)
    var = rpc.call("com.apple.instruments.server.services.sysmontap", "start").selector
    # print(f"start {var}")
    time.sleep(0.5)
    # 等待回调触发并获取结果
    try:
        result = result_queue.get(timeout=5)  # 等待 5 秒
    except queue.Empty:
        print("Timeout: No message received from on_sysmontap_message.")
        result = None

    var = rpc.call("com.apple.instruments.server.services.sysmontap", "stop").selector
    # print(f"stop {var}")
    rpc.stop()
    return result


#
# def sysmontap(rpc):
#     result_queue = queue.Queue()
#     def dropped_message(res):
#         print("[DROP]", res.selector, res.raw.channel_code)
#
#     def on_sysmontap_message(res):
#         if isinstance(res.selector, list):            # return json.dumps(res.selector)
#             # print(json.dumps(res.selector, indent=4))
#             result_queue.put(res.selector)
#
#
#     rpc.register_undefined_callback(dropped_message)
#     config = {
#         "bm": 0,
#         "cpuUsage": True,
#         "procAttrs": [
#             "memVirtualSize", "cpuUsage", "ctxSwitch", "intWakeups",
#             "physFootprint", "memResidentSize", "memAnon", "pid"
#         ],
#         "sampleInterval": 1000000000,  # 1e9 ns == 1s
#         "sysAttrs": [
#             "vmExtPageCount", "vmFreeCount", "vmPurgeableCount",
#             "vmSpeculativeCount", "physMemSize"
#         ],
#         "ur": 1000
#     }
#     rpc.call("com.apple.instruments.server.services.sysmontap", "setConfig:", config)
#     rpc.register_channel_callback("com.apple.instruments.server.services.sysmontap", on_sysmontap_message)
#     var = rpc.call("com.apple.instruments.server.services.sysmontap", "start").selector
#     print(f"start {var}")
#     # time.sleep(0.5)
#     # 等待回调触发并获取结果
#     try:
#         # 设置超时时间，避免无限等待
#         result = result_queue.get(timeout=5)  # 等待 5 秒
#     except queue.Empty:
#         print("Timeout: No message received from on_sysmontap_message.")
#         result = None
#     var = rpc.call("com.apple.instruments.server.services.sysmontap", "stop").selector
#     print(f"stop {var}")
#     rpc.stop()
#     return result


# class SysMonTap:
#     def __init__(self, rpc):
#         self.rpc = rpc
#         self.result_queue = queue.Queue()
#         self.is_initialized = False  # 标志是否已完成初始化
#         self.is_running = False  # 标志是否已启动 sysmontap
#
#     def initialize(self):
#         """初始化 sysmontap 配置和回调，只执行一次"""
#         if self.is_initialized:
#             return  # 如果已经初始化，则直接返回
#
#         def dropped_message(res):
#             print("[DROP]", res.selector, res.raw.channel_code)
#
#         def on_sysmontap_message(res):
#             if isinstance(res.selector, list):
#                 self.result_queue.put(res.selector)
#
#         # 注册回调
#         self.rpc.register_undefined_callback(dropped_message)
#         self.rpc.register_channel_callback(
#             "com.apple.instruments.server.services.sysmontap",
#             on_sysmontap_message
#         )
#
#         # 配置 sysmontap
#         config = {
#             "bm": 0,
#             "cpuUsage": True,
#             "procAttrs": [
#                 "memVirtualSize", "cpuUsage", "ctxSwitch", "intWakeups",
#                 "physFootprint", "memResidentSize", "memAnon", "pid"
#             ],
#             "sampleInterval": 1000000000,  # 1e9 ns == 1s
#             "sysAttrs": [
#                 "vmExtPageCount", "vmFreeCount", "vmPurgeableCount",
#                 "vmSpeculativeCount", "physMemSize"
#             ],
#             "ur": 1000
#         }
#         self.rpc.call("com.apple.instruments.server.services.sysmontap", "setConfig:", config)
#
#         self.is_initialized = True  # 标记为已初始化
#
#     def start(self):
#         """启动 sysmontap，只执行一次"""
#         # 确保已初始化
#         self.initialize()
#
#         if self.is_running:
#             print("SysMonTap is already running.")
#             return  # 如果已经启动，则直接返回
#
#         # 启动 sysmontap
#         var = self.rpc.call("com.apple.instruments.server.services.sysmontap", "start").selector
#         print(f"start {var}")
#         self.is_running = True  # 标记为已启动
#
#     def get_result(self, timeout=5):
#         """从队列中获取结果"""
#         try:
#             # 设置超时时间，避免无限等待
#             result = self.result_queue.get(timeout=timeout)  # 等待指定时间
#             return result
#         except queue.Empty:
#             print("Timeout: No message received from on_sysmontap_message.")
#             return None
#
#     def stop(self):
#         """停止 sysmontap 服务"""
#         if not self.is_running:
#             print("SysMonTap is not running.")
#             return  # 如果未启动，则直接返回
#
#         var = self.rpc.call("com.apple.instruments.server.services.sysmontap", "stop").selector
#         print(f"stop {var}")
#         self.rpc.stop()
#         self.is_running = False  # 标记为已停止