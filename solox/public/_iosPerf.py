#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Created on Tue May 11 2021 16:30:17 by codeskyblue
"""

import enum
import queue
import threading
import time
import typing
from collections import defaultdict, namedtuple
from typing import Any, Iterator, Optional, Tuple, Union
import weakref
import pdb
from multiprocessing import Process, Queue

from solox.public.demo.instrument_demo.gpu import gpu
from solox.public.demo.instrument_demo.graphics import cmd_graphics
from solox.public.demo.instrument_demo.networking import networking
from solox.public.demo.instrument_demo.sysmontap import sysmontap
from solox.public.iosperf._device import BaseDevice
from solox.public.iosperf._proto import *
from ios_device.remote.remote_lockdown import RemoteLockdownClient
from ios_device.servers.Instrument import InstrumentServer
from ios_device.cli.base import InstrumentsBase
from typing import Dict, Any
from ios_device.util.utils import DumpDisk, DumpNetwork, DumpMemory, convertBytes, \
    MOVIE_FRAME_COST, NANO_SECOND, kperf_data


class DataType(str, enum.Enum):
    SCREENSHOT = "screenshot"
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"  # 流量
    FPS = "fps"
    PAGE = "page"
    GPU = "gpu"


CallbackType = typing.Callable[[DataType, dict], None]

_singleton_lock = threading.Lock()
_singleton_result = None  # 存储结果
_singleton_executed = False  # 标记是否已经执行过


class RunningProcess:
    """ acturally there is a better way to monitor process pid """
    PID_UPDATE_DURATION = 5.0

    def __init__(self, d: BaseDevice, bundle_id: str):
        self._version_compare = -1
        if self.compare_version_lists(d.device_info()["ProductVersion"], "17.0") >= 0:
            self._version_compare = 1
        else:
            self._ins = d.connect_instruments()
            self._bundle_id = bundle_id
            self._app_infos = list(d.installation.iter_installed(app_type=None))
            self._next_update_time = 0.0
            self._last_pid = None
            self._lock = threading.Lock()
            weakref.finalize(self, self._ins.close)

    def get_version_lists(self):
        return self._version_compare

    def compare_version_lists(self, v1_lst, v2_lst):
        """
        比较版本号列表，从高位到底位逐位比较，根据情况判断大小。
        :param v1_lst:
        :param v2_lst:
        :return:
        """
        for v1, v2 in zip(v1_lst, v2_lst):
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
        return 0

    @property
    def bundle_id(self) -> str:
        return self._bundle_id

    def get_pid(self) -> Union[int, None]:
        """ return pid """
        with self._lock:
            if time.time() < self._next_update_time:
                return self._last_pid

            if self._last_pid and self._ins.is_running_pid(self._last_pid):
                self._next_update_time = time.time() + self.PID_UPDATE_DURATION
                return self._last_pid

            for pinfo in self._ins.app_process_list(self._app_infos):
                if pinfo['bundle_id'] == self._bundle_id:
                    self._last_pid = pinfo['pid']
                    self._next_update_time = time.time(
                    ) + self.PID_UPDATE_DURATION
                    # print(self._bundle_id, "pid:", self._last_pid)
                    return self._last_pid


class WaitGroup(object):
    """WaitGroup is like Go sync.WaitGroup.

    Without all the useful corner cases.
    """

    def __init__(self):
        self.count = 0
        self.cv = threading.Condition()

    def add(self, n):
        self.cv.acquire()
        self.count += n
        self.cv.release()

    def done(self):
        self.cv.acquire()
        self.count -= 1
        if self.count == 0:
            self.cv.notify_all()
        self.cv.release()

    # FIXME(ssx): here should quit when timeout, but maybe not
    def wait(self, timeout: Optional[float] = None):
        self.cv.acquire()
        while self.count > 0:
            self.cv.wait(timeout=timeout)
        self.cv.release()


def gen_stimestamp(seconds: Optional[float] = None) -> str:
    """ 生成专门用于tmq-service.taobao.org平台使用的timestampString """
    if seconds is None:
        seconds = time.time()
    return int(seconds * 1000)


def iter_fps(d: BaseDevice, rp: RunningProcess, address=None, rsdPort=None, lock=None) -> Iterator[Any]:
    index = 0
    if rp.get_version_lists() == 1:
        host = address
        port = rsdPort
        with lock:
            rsd_1 = RemoteLockdownClient((host, port), userspace_port=60106)

            rsd_1.connect()
            rpc = InstrumentServer(rsd_1).init()
            result_queue = queue.Queue()
            data = cmd_graphics(rpc, result_queue)
            rsd_1.close()
            rpc.stop()
        fps = data['CoreAnimationFramesPerSecond']  # fps from GPU
        yield DataType.FPS, {"fps": fps, "time": time.time(), "value": fps}
    else:
        with d.instruments_context() as ts:
            for data in ts.iter_opengl_data():
                if index != 1:
                    index = index + 1
                    continue
                fps = data['CoreAnimationFramesPerSecond']  # fps from GPU
                yield DataType.FPS, {"fps": fps, "time": time.time(), "value": fps}


def iter_gpu(d: BaseDevice, rp: RunningProcess, address=None, rsdPort=None, lock=None) -> Iterator[Any]:
    index = 0
    if rp.get_version_lists() == 1:
        host = address
        port = rsdPort
        with lock:
            rsd_1 = RemoteLockdownClient((host, port), userspace_port=60106)
            rsd_1.connect()
            rpc = InstrumentServer(rsd_1).init()
            result_queue = queue.Queue()
            data = cmd_graphics(rpc, result_queue)
            rsd_1.close()
            rpc.stop()
        device_utilization = data['Device Utilization %']  # Device Utilization
        tiler_utilization = data['Tiler Utilization %']  # Tiler Utilization
        renderer_utilization = data['Renderer Utilization %']  # Renderer Utilization
        yield DataType.GPU, {"device": device_utilization, "renderer": renderer_utilization,
                         "tiler": tiler_utilization, "time": time.time(), "value": device_utilization}
    else:
        with d.instruments_context() as ts:
            for data in ts.iter_opengl_data():
                device_utilization = data['Device Utilization %']  # Device Utilization
                tiler_utilization = data['Tiler Utilization %']  # Tiler Utilization
                renderer_utilization = data['Renderer Utilization %']  # Renderer Utilization
                yield DataType.GPU, {"device": device_utilization, "renderer": renderer_utilization,
                                     "tiler": tiler_utilization, "time": time.time(), "value": device_utilization}


def iter_screenshot(d: BaseDevice) -> Iterator[Tuple[DataType, dict]]:
    for img in d.iter_screenshot():
        _time = time.time()
        img.thumbnail((500, 500))  # 缩小图片已方便保存

        # example of convert image to bytes
        # buf = io.BytesIO()
        # img.save(buf, format="JPEG")

        # turn image to URL
        yield DataType.SCREENSHOT, {"time": _time, "value": img}


ProcAttrs = namedtuple("ProcAttrs", SYSMON_PROC_ATTRS)


def execute_singleton(host: str, port: int, log: str = "cpu") -> dict:
    """
    只能被单线程执行一次，其他线程通过返回值获取结果。
    """
    print("execute_singleton:first")
    global _singleton_result, _singleton_executed
    with _singleton_lock:
        if _singleton_executed:
            # 如果已经执行过，直接 yield 结果
            print("已经被执行,当前线程：", threading.current_thread())
            yield _singleton_result
            return

        # 如果未执行过，开始执行
        print("未被执行，首次执行，当前线程：", threading.current_thread())
        _singleton_executed = True

        # 模拟 RemoteLockdownClient 的上下文管理
        with RemoteLockdownClient((host, port), userspace_port=60106) as rsd:
            print("Executing:", threading.current_thread())

            rpc = InstrumentServer(rsd).init()
            result_queue = queue.Queue()
            result = sysmontap(rpc, result_queue)
            print("Result fetched:", threading.current_thread())
            _singleton_result = result
    # #         rpc.stop()
    # #         pid = 11727
    # #         info = result
    # #         # print(info)
    # #         # if info is None or len(info) != 2:
    # #         #         continue
    # #         sinfo, pinfolist = info
    # #         if 'CPUCount' not in sinfo:
    # #             sinfo, pinfolist = pinfolist, sinfo
    # #
    # #         # if 'CPUCount' not in sinfo:
    # #         #     continue
    # #
    # #         cpu_count = sinfo['CPUCount']
    # #
    # #         sys_cpu_usage = sinfo['SystemCPUUsage']
    # #         cpu_total_load = sys_cpu_usage['CPU_TotalLoad']
    # #         cpu_user = sys_cpu_usage['CPU_UserLoad']
    # #         cpu_sys = sys_cpu_usage['CPU_SystemLoad']
    # #
    # #         # if 'Processes' not in pinfolist:
    # #         #     continue
    # #
    # #         # 这里的total_cpu_usage加起来的累计值大概在0.5~5.0之间
    # #         total_cpu_usage = 0.0
    # #         for attrs in pinfolist['Processes'].values():
    # #             pinfo = ProcAttrs(*attrs)
    # #             if isinstance(pinfo.cpuUsage, float):  # maybe NSNull
    # #                 total_cpu_usage += pinfo.cpuUsage
    # #
    # #         cpu_usage = 0.0
    # #         attrs = pinfolist['Processes'].get(pid)
    # #         if attrs is None:  # process is not running
    # #             # continue
    # #             # print('process not launched')
    # #             pass
    # #         else:
    # #             assert len(attrs) == len(SYSMON_PROC_ATTRS)
    # #             # print(ProcAttrs, attrs)
    # #             pinfo = ProcAttrs(*attrs)
    # #             if pinfo.cpuUsage is not None:
    # #                 cpu_usage = pinfo.cpuUsage
    # #         print(dict(
    # #             type="process",
    # #             pid=pid,
    # #             phys_memory=pinfo.physFootprint,  # 物理内存
    # #             phys_memory_string="{:.1f} MiB".format(pinfo.physFootprint / 1024 /
    # #                                                    1024),
    # #             vss=pinfo.memVirtualSize,
    # #             rss=pinfo.memResidentSize,
    # #             anon=pinfo.memAnon,  # 匿名内存? 这个是啥
    # #             cpu_count=cpu_count,
    # #             cpu_usage=cpu_usage,  # 理论上最高 100.0 (这里是除以过cpuCount的)
    # #             sys_cpu_usage=cpu_total_load,
    # #             attr_cpuUsage=pinfo.cpuUsage,
    # #             attr_cpuTotal=cpu_total_load,
    # #             attr_ctxSwitch=pinfo.ctxSwitch,
    # #             attr_intWakeups=pinfo.intWakeups,
    # #             attr_systemInfo=sys_cpu_usage))
    # # yield dict(
    # #     type="process",
    # #     pid=pid,
    # #     phys_memory=pinfo.physFootprint,  # 物理内存
    # #     phys_memory_string="{:.1f} MiB".format(pinfo.physFootprint / 1024 /
    # #                                            1024),
    # #     vss=pinfo.memVirtualSize,
    # #     rss=pinfo.memResidentSize,
    # #     anon=pinfo.memAnon,  # 匿名内存? 这个是啥
    # #     cpu_count=cpu_count,
    # #     cpu_usage=cpu_usage,  # 理论上最高 100.0 (这里是除以过cpuCount的)
    # #     sys_cpu_usage=cpu_total_load,
    # #     attr_cpuUsage=pinfo.cpuUsage,
    # #     attr_cpuTotal=cpu_total_load,
    # #     attr_ctxSwitch=pinfo.ctxSwitch,
    # #     attr_intWakeups=pinfo.intWakeups,
    # #     attr_systemInfo=sys_cpu_usage)
    # # 构造返回值
    _singleton_result = dict(
        type="process",
        pid=65165,
        phys_memory=0,  # 物理内存
        phys_memory_string="0 MiB",
        vss=0,
        rss=0,
        anon=0,  # 匿名内存
        cpu_count=0,
        cpu_usage=0,  # 理论上最高 100.0 (这里是除以过cpuCount的)
        sys_cpu_usage=0,
        attr_cpuUsage=0.0,
        attr_cpuTotal=0,
        attr_ctxSwitch=0.0,
        attr_intWakeups=0.0,
        attr_systemInfo=0
    )

    # 使用 yield 返回结果
    return _singleton_result


def worker(queue):
    host, port = queue.get()
    with RemoteLockdownClient((host, port), userspace_port=60106) as rsd:
        print(rsd.product_version)

def _iter_complex_cpu_memory(d: BaseDevice,
                             rp: RunningProcess, address=None, rsdPort=None, lock=None, pid=None) -> Iterator[dict]:
    """
    content in iterator

    - {'type': 'system_cpu',
        'sys': -1.0,
        'total': 55.21212121212122,
        'user': -1.0}
    - {'type': 'process',
        'cpu_usage': 2.6393411792622925,
        'mem_anon': 54345728,
        'mem_rss': 130760704,
        'pid': 1344}
    """
    if rp.get_version_lists() == 1:
        host = address
        port = rsdPort
        with lock:
            rsd_1 = RemoteLockdownClient((host, port), userspace_port=60106)
            rsd_1.connect()
            rpc = InstrumentServer(rsd_1).init()
            result_queue = queue.Queue()
            result = sysmontap(rpc, result_queue)
            rsd_1.close()
            rpc.stop()
        pid = pid
        info = result
        # print(info)
        # if info is None or len(info) != 2:
        #         continue
        sinfo, pinfolist = info
        if 'CPUCount' not in sinfo:
            sinfo, pinfolist = pinfolist, sinfo

        # if 'CPUCount' not in sinfo:
        #     continue

        cpu_count = sinfo['CPUCount']

        sys_cpu_usage = sinfo['SystemCPUUsage']
        cpu_total_load = sys_cpu_usage['CPU_TotalLoad']
        cpu_user = sys_cpu_usage['CPU_UserLoad']
        cpu_sys = sys_cpu_usage['CPU_SystemLoad']
        # if 'Processes' not in pinfolist:
        #     continue
        # 这里的total_cpu_usage加起来的累计值大概在0.5~5.0之间
        total_cpu_usage = 0.0
        for attrs in pinfolist['Processes'].values():
            pinfo = ProcAttrs(*attrs)
            if isinstance(pinfo.cpuUsage, float):  # maybe NSNull
                total_cpu_usage += pinfo.cpuUsage

        cpu_usage = 0.0
        attrs = pinfolist['Processes'].get(pid)

        if attrs is None:  # process is not running
            # continue
            # print('process not launched')
            pass
        else:
            assert len(attrs) == len(SYSMON_PROC_ATTRS)
            # print(ProcAttrs, attrs)
            pinfo = ProcAttrs(*attrs)
            if pinfo.cpuUsage is not None:
                cpu_usage = pinfo.cpuUsage

        yield dict(
            type="process",
            pid=pid,
            phys_memory=pinfo.physFootprint,  # 物理内存
            phys_memory_string="{:.1f} MiB".format(pinfo.physFootprint / 1024 /
                                                   1024),
            vss=pinfo.memVirtualSize,
            rss=pinfo.memResidentSize,
            anon=pinfo.memAnon,  # 匿名内存? 这个是啥
            cpu_count=cpu_count,
            cpu_usage=cpu_usage,  # 理论上最高 100.0 (这里是除以过cpuCount的)
            sys_cpu_usage=cpu_total_load,
            attr_cpuUsage=pinfo.cpuUsage,
            attr_cpuTotal=cpu_total_load,
            attr_ctxSwitch=pinfo.ctxSwitch,
            attr_intWakeups=pinfo.intWakeups,
            attr_systemInfo=sys_cpu_usage)

    else:
        with d.instruments_context() as ts:
            for info in ts.iter_cpu_memory():
                pid = rp.get_pid()
                if info is None or len(info) != 2:
                    continue
                sinfo, pinfolist = info
                if 'CPUCount' not in sinfo:
                    sinfo, pinfolist = pinfolist, sinfo

                if 'CPUCount' not in sinfo:
                    continue

                cpu_count = sinfo['CPUCount']

                sys_cpu_usage = sinfo['SystemCPUUsage']
                cpu_total_load = sys_cpu_usage['CPU_TotalLoad']
                cpu_user = sys_cpu_usage['CPU_UserLoad']
                cpu_sys = sys_cpu_usage['CPU_SystemLoad']

                if 'Processes' not in pinfolist:
                    continue

                # 这里的total_cpu_usage加起来的累计值大概在0.5~5.0之间
                total_cpu_usage = 0.0
                for attrs in pinfolist['Processes'].values():
                    pinfo = ProcAttrs(*attrs)
                    if isinstance(pinfo.cpuUsage, float):  # maybe NSNull
                        total_cpu_usage += pinfo.cpuUsage

                cpu_usage = 0.0
                attrs = pinfolist['Processes'].get(pid)
                if attrs is None:  # process is not running
                    # continue
                    # print('process not launched')
                    pass
                else:
                    assert len(attrs) == len(SYSMON_PROC_ATTRS)
                    # print(ProcAttrs, attrs)
                    pinfo = ProcAttrs(*attrs)
                    cpu_usage = pinfo.cpuUsage
                # next_list_process_time = time.time() + next_timeout
                # cpu_usage, rss, mem_anon, pid = pinfo

                # 很诡异的计算方法，不过也就这种方法计算出来的CPU看起来正常一点
                # 计算后的cpuUsage范围 [0, 100]
                # cpu_total_load /= cpu_count
                # cpu_usage *= cpu_total_load
                # if total_cpu_usage > 0:
                #     cpu_usage /= total_cpu_usage

                # print("cpuUsage: {}, total: {}".format(cpu_usage, total_cpu_usage))
                # print("memory: {} MB".format(pinfo.physFootprint / 1024 / 1024))
                yield dict(
                    type="process",
                    pid=pid,
                    phys_memory=pinfo.physFootprint,  # 物理内存
                    phys_memory_string="{:.1f} MiB".format(pinfo.physFootprint / 1024 /
                                                           1024),
                    vss=pinfo.memVirtualSize,
                    rss=pinfo.memResidentSize,
                    anon=pinfo.memAnon,  # 匿名内存? 这个是啥
                    cpu_count=cpu_count,
                    cpu_usage=cpu_usage,  # 理论上最高 100.0 (这里是除以过cpuCount的)
                    sys_cpu_usage=cpu_total_load,
                    attr_cpuUsage=pinfo.cpuUsage,
                    attr_cpuTotal=cpu_total_load,
                    attr_ctxSwitch=pinfo.ctxSwitch,
                    attr_intWakeups=pinfo.intWakeups,
                    attr_systemInfo=sys_cpu_usage)


def iter_memory(d: BaseDevice, rp: RunningProcess, address=None, rsdPort=None, lock=None, pid=None) -> Iterator[Any]:
    for minfo in _iter_complex_cpu_memory(d, rp, address, rsdPort, lock, pid):  # d.iter_cpu_mem(bundle_id):
        yield DataType.MEMORY, {
            "pid": minfo['pid'],
            "timestamp": gen_stimestamp(),
            "value": minfo['phys_memory'] / 1024 / 1024,  # MB
        }


def iter_cpu(d: BaseDevice, rp: RunningProcess, address=None, rsdPort=None, lock=None, pid=None) -> Iterator[Any]:
    try:
        for minfo in _iter_complex_cpu_memory(d, rp, address, rsdPort, lock, pid):  # d.iter_cpu_mem(bundle_id):
            yield DataType.CPU, {
                "timestamp": gen_stimestamp(),
                "pid": minfo['pid'],
                "value": minfo['cpu_usage'],  # max 100.0?, maybe not
                "sys_value": minfo['sys_cpu_usage'],
                "count": minfo['cpu_count']
            }
    except Exception as e:
        print(e)


def set_interval(it: Iterator[Any], interval: float):
    while True:
        start = time.time()
        data = next(it)
        yield data
        wait = max(0, interval - (time.time() - start))
        time.sleep(wait)


def iter_network_flow(d: BaseDevice, rp: RunningProcess, address=None, rsdPort=None) -> Iterator[Any]:
    n = 0
    if rp.get_version_lists() == 1:
        host = address
        port = rsdPort
        # with lock:
        rsd_1 = RemoteLockdownClient((host, port), userspace_port=60106)
        rsd_1.connect()
        rpc = InstrumentServer(rsd_1).init()
        result_queue = queue.Queue()
        data = networking(rpc, result_queue)
        rsd_1.close()
        rpc.stop()
    else:
        with d.connect_instruments() as ts:
            for nstat in ts.iter_network():
                # if n < 10:
                #     n += 1
                #     continue
                yield DataType.NETWORK, {
                    "timestamp": gen_stimestamp(),
                    "downFlow": (nstat['rx.bytes'] or 0) / 1024,
                    "upFlow": (nstat['tx.bytes'] or 0) / 1024
                }


def append_data(wg: WaitGroup, stop_event: threading.Event,
                idata: Iterator[Any], callback: CallbackType, filters: list):

    for _type, data in idata:
        assert isinstance(data, dict)
        assert isinstance(_type, DataType)

        if stop_event.is_set():
            wg.done()
            break

        if isinstance(data, dict) and "time" in data:
            stimestamp = gen_stimestamp(data.pop('time'))
            data.update({"timestamp": stimestamp})

        # result[_type].append(data)
        if _type in filters:
            callback(_type, data)
        if data:
            if _type.value == 'network':
                return data['downFlow'], data['upFlow']
            elif _type.value == 'cpu':
                app_cpu = data['value']
                sys_cpu = data['sys_value']
                if data['count'] > 0:
                    app_cpu /= data['count']
                    sys_cpu /= data['count']
                print(app_cpu, " 00000", sys_cpu)
                return app_cpu, sys_cpu
            else:
                return data['value']
        # print(_type, data)

    stop_event.set()  # 当有一个中断，其他的全部中断，让错误暴露出来


class Performance():
    # PROMPT_TITLE = "tidevice performance"

    def __init__(self, d: BaseDevice, perfs: typing.List[DataType] = [], address=None, rsdPort=None):
        self._rp = None
        self._d = d
        self._bundle_id = None
        self._stop_event = threading.Event()
        self._wg = WaitGroup()
        self._started = False
        self._result = defaultdict(list)
        self._perfs = perfs
        self._address = address
        self._rsdPort = rsdPort

        # the callback function accepts all the data
        self._callback = None

    def start(self, bundle_id: str, callback: CallbackType = None, lock=None, pid=None):
        _perfValue = 0
        if not callback:
            # 默认不输出屏幕的截图（暂时没想好怎么处理）
            callback = lambda _type, data: print(_type.value, data,
                                                 flush=True) if _type != DataType.SCREENSHOT and _type in self._perfs else None

        self._rp = RunningProcess(self._d, bundle_id)

        _perfValue = self._thread_start(callback, lock=lock, pid=pid)
        return _perfValue

    def _thread_start(self, callback: CallbackType, lock=None, pid=None):
        iters = []
        _perfValue = 0

        print(self._perfs)
        if DataType.CPU in self._perfs:
            iters.append(iter_cpu(self._d, self._rp, self._address, self._rsdPort, lock, pid))
        if DataType.MEMORY in self._perfs:
            iters.append(iter_memory(self._d, self._rp, self._address, self._rsdPort, lock, pid))
        if DataType.FPS in self._perfs:
            iters.append(iter_fps(self._d, self._rp, self._address, self._rsdPort, lock))
        if DataType.GPU in self._perfs:
            iters.append(iter_gpu(self._d, self._rp, self._address, self._rsdPort, lock))
        if DataType.SCREENSHOT in self._perfs:
            iters.append(set_interval(iter_screenshot(self._d), 1.0))
        if DataType.NETWORK in self._perfs:
            iters.append(iter_network_flow(self._d, self._rp, self._address, self._rsdPort))
        for it in (iters):  # yapf: disable
            self._wg.add(1)
            _perfValue = append_data(self._wg, self._stop_event, it, callback, self._perfs)
            break

        print(_perfValue)
        return _perfValue

    def stop(self):  # -> PerfReport:
        self._stop_event.set()
        # memory and fps will take at least 1 second to catch _stop_event
        # to make function run faster, we not using self._wg.wait(..) here
        # > self._wg.wait(timeout=3.0) # wait all stopped
        # > self._started = False

    def wait(self, timeout: float):
        return self._wg.wait(timeout=timeout)
