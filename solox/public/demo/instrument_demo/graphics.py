"""
graphics 计算 fps
"""
import queue
import time
from ios_device.servers.Instrument import InstrumentServer


def cmd_graphics(rpc, result_queue):
    last_timestamp = 0

    def dropped_message(res):
        print("[DROP]", res.selector, res.raw.channel_code)

    def on_graphics_message(res):
        data = res.selector
        # nonlocal last_timestamp
        # cost_time = data['XRVideoCardRunTimeStamp'] - last_timestamp
        # last_timestamp = data['XRVideoCardRunTimeStamp']
        # print(cost_time,'fps-----:', data['CoreAnimationFramesPerSecond'])
        result_queue.put(data)

    rpc.register_undefined_callback(dropped_message)
    rpc.register_channel_callback("com.apple.instruments.server.services.graphics.opengl", on_graphics_message)
    rpc.call("com.apple.instruments.server.services.graphics.opengl", "startSamplingAtTimeInterval:", 0.0)
    time.sleep(0.5)
    try:
        result = result_queue.get(timeout=5)  # 等待 5 秒
    except queue.Empty:
        print("Timeout: No message received from on_sysmontap_message.")
        result = None
    rpc.call("com.apple.instruments.server.services.graphics.opengl", "stopSampling")
    rpc.stop()
    return result


if __name__ == '__main__':
    rpc = InstrumentServer().init()
    cmd_graphics(rpc)
    rpc.stop()
