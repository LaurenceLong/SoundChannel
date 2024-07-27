import itertools
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time

import numpy as np
import pyaudio
from amodem import audio
from amodem import common
from amodem import framing
from amodem import sampling
from amodem.__main__ import _config_log
from amodem.async_reader import AsyncReader
from amodem.config import bitrates, Configuration
from amodem.detect import Detector
from amodem.framing import Framer
from amodem.recv import Receiver
from amodem.send import Sender
from amodem.stream import Reader
from enum import Enum

log = logging.getLogger("__name__")


class _tmp:
    verbose = 0
    quiet = False


_config_log(_tmp())

KEY_NAME = "name"
KEY_MSG = "msg"
KEY_SIZE = "size"
KEY_QUIET = "is_quiet"

VAL_MSG_NAME = "\\msg_str/"

NEGOT_START = "negotiate_start~!@#$%^&*()_+;"
NEGOT_SNR = "negotiate_snr~!@#$%^&*()_+;"
NEGOT_END = "negotiate_end~!@#$%^&*()_+;"

FORMAT = pyaudio.paInt16
CHANNELS = 1
INIT_SEND_kbps = 80
INIT_RECV_kbps = 80


def create_config(kbps):
    config = bitrates[kbps]
    config.silence_start = config.silence_stop = 0.15
    config.timeout = float("inf")
    return config


def get_application_path():
    if getattr(sys, "frozen", False):
        # 如果是打包后的应用程序
        return os.path.dirname(sys.executable)
    else:
        # 如果是在开发环境中运行
        return os.path.dirname(os.path.abspath(__file__))


CWD = get_application_path()


class Evt(Enum):
    SPEND_TIME = "SEND_TIME"
    SEND_FILE_START = "SEND_FILE_START"
    SEND_FINISH = "SEND_FINISH"
    RECV_FILE_START = "RECV_FILE_START"
    RECV_FILE_FINISH = "RECV_FILE_FINISH"
    NOTIFY_MSG = "NOTIFY_MSG"
    NOTIFY_FILE = "NOTIFY_FILE"
    RE_TRANSMIT = "RE_TRANSMIT"


class Event:
    def __init__(self, key, value, o1=None, o2=None, o3=None):
        self.key = key
        self.value = value
        self.o1 = o1
        self.o2 = o2
        self.o3 = o3


class WrappedData:

    def __init__(self, data_name, data_content, quiet=False):
        self.data_name = data_name
        self.data_content = data_content
        self.quiet = quiet
        if isinstance(data_content, str):
            self.data_size = len(data_content.encode("utf-8"))
        else:
            self.data_size = len(data_content)

    def get_name(self):
        return self.data_name

    def get_data(self):
        return self.data_content

    def get_size(self):
        return self.data_size

    def is_quiet(self):
        return self.quiet


class SoundChannelBase:

    def __init__(self, ):
        self.recv_cfg = create_config(INIT_SEND_kbps)
        self.send_cfg = create_config(INIT_RECV_kbps)
        self.negot_cfg = Configuration(Fs=32e3, Npoints=16, frequencies=self.send_cfg.negotiate_frequencies)
        self.negot_cfg.timeout = float("inf")
        self.negot_cfg.kbps = 4

        self.data_interface = self.create_interface(self.send_cfg)
        # self.negot_interface = self.create_interface(self.negot_cfg)

        self.opened_streams = []
        self.r_stream = self.create_recv_stream(self.data_interface)
        self.s_stream = self.create_send_stream(self.data_interface)
        # self.negot_r_stream = self.create_recv_stream(self.negot_interface)
        # self.negot_s_stream = self.create_send_stream(self.negot_interface)
        self.opened_streams.append(self.r_stream)
        self.opened_streams.append(self.s_stream)
        # self.opened_streams.append(self.negot_r_stream)
        # self.opened_streams.append(self.negot_s_stream)

        self.listening = True

        self.notify_event_queue = queue.Queue()  # notify to ui
        self.recv_event_queue = queue.Queue()  # recv event to ui
        self.send_event_queue = queue.Queue()  # send event to ui
        self.negot_event_queue = queue.Queue()  # send negotiate event to ui

        self.data_task_queue = queue.Queue()
        self.negot_task_queue = queue.Queue()

        self.ready_events = [threading.Event() for _ in range(2)]
        self.stop_events = [threading.Event() for _ in range(2)]  # todo 协商线程加上需改成4
        self.running_threads = []
        self.p_listen = threading.Thread(
            target=self.listen_loop,
            args=(self.load_r_stream, self.load_recv_cfg, self.ready_events[0], self.stop_events[0],
                  self.recv_event_queue,))
        self.p_send = threading.Thread(
            target=self.send_loop,
            args=(self.load_s_stream, self.load_send_cfg, self.data_task_queue, self.ready_events[1],
                  self.stop_events[1], self.send_event_queue,))
        # self.negot_listen = threading.Thread(
        #     target=self.listen_loop,
        #     args=(self.load_negot_r_stream, self.load_negot_cfg, self.ready_events[2], self.stop_events[2],
        #           self.negot_event_queue,))
        # self.negot_send = threading.Thread(
        #     target=self.send_loop,
        #     args=(self.load_negot_s_stream, self.load_negot_cfg, self.negot_task_queue, self.ready_events[3],
        #           self.stop_events[3], self.negot_event_queue,))
        self.running_threads.append(self.p_listen)
        self.running_threads.append(self.p_send)
        # self.running_threads.append(self.negot_listen)
        # self.running_threads.append(self.negot_send)
        for th in self.running_threads:
            th.daemon = True
            th.start()
        for ready_evt in self.ready_events:
            ready_evt.wait()

    def load_r_stream(self):
        return self.r_stream

    def load_s_stream(self):
        return self.s_stream

    def load_negot_r_stream(self):
        return self.negot_r_stream

    def load_negot_s_stream(self):
        return self.negot_s_stream

    def load_recv_cfg(self):
        return self.recv_cfg

    def load_send_cfg(self):
        return self.send_cfg

    def load_negot_cfg(self):
        return self.negot_cfg

    def create_interface(self, cfg):
        interface = audio.Interface(cfg)
        interface.load("")
        return interface

    def create_send_stream(self, interface):
        return interface.player()

    def create_recv_stream(self, interface):
        recv_stream = interface.recorder()
        return AsyncReader(recv_stream, recv_stream.bufsize)

    def send_message(self, msg):
        if isinstance(msg, str):
            self.queue_send_data(self.data_task_queue, msg)

    def send_file(self, file_path):
        _, file_name = os.path.split(file_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            with open(file_path, "rb") as fd:
                data = fd.read()
                self.queue_send_data(self.data_task_queue, data, filename=file_name)

    def queue_send_data(self, task_queue, data, filename=None):
        if filename:
            task_queue.put(WrappedData(filename, data))
        else:
            task_queue.put(WrappedData(VAL_MSG_NAME, data))

    def receive_data_signal(self, stream, config, dst=None, stop_event: threading.Event = None):
        reader = Reader(stream, data_type=common.loads)
        signal = itertools.chain.from_iterable(reader)

        pylab = common.Dummy()
        detector = Detector(config=config, pylab=pylab)
        receiver = Receiver(config=config, pylab=pylab)
        dst = dst or tempfile.TemporaryFile()
        try:
            log.info("Waiting for carrier tone: %.1f kHz" % (config.Fc / 1e3))
            signal, amplitude, freq_error = detector.run(signal, stop_event=stop_event)

            freq = 1 / (1.0 + freq_error)  # receiver"s compensated frequency
            log.debug("Frequency correction: %.3f ppm" % ((freq - 1) * 1e6))

            gain = 1.0 / amplitude
            log.debug("Gain correction: %.3f" % gain)

            sampler = sampling.Sampler(signal,
                                       sampling.defaultInterpolator,
                                       freq=freq)
            receiver.run(sampler, gain=1.0 / amplitude, output=dst)
            return dst
        except BaseException:  # pylint: disable=broad-except
            import traceback
            traceback.print_exc()
            log.error("Decoding failed")
            return False
        finally:
            dst.flush()
            dst.seek(0)
            receiver.report()

    def send_data_bytes(self, stream, config, bytes_data, stop_event: threading.Event, send_event_queue: queue.Queue):
        t0 = time.time()
        send_time = config.silence_start + config.silence_stop + len(bytes_data) * 8 / config.modem_bps
        send_event_queue.put(Event(Evt.SPEND_TIME, send_time))

        sender = Sender(stream, config=config)
        # pre-padding audio with silence (priming the audio sending queue)
        sender.write(np.zeros(int(config.Fs * config.silence_start)))
        sender.start()

        training_duration = sender.offset
        log.info("Sending %.3f seconds of training audio" % (training_duration / config.Fs))
        framer = Framer()
        bits = framing.encode(bytes_data, framer=framer)
        log.info("Starting modulation")
        sender.modulate(bits=bits, stop_event=stop_event)

        data_duration = sender.offset - training_duration
        log.info("Sent %.3f kB @ %.3f seconds" % (len(bytes_data) / 1e3, data_duration / config.Fs))

        # post-padding audio with silence
        sender.write(np.zeros(int(config.Fs * config.silence_stop)))
        t1 = time.time()
        log.info("Total Used time %.4f seconds" % (t1 - t0))

        send_event_queue.put(Event(Evt.SEND_FINISH, ""))
        return

    def send_handshake(self, stream, config, wrapped_data: WrappedData, stop_event: threading.Event,
                       send_event_queue: queue.Queue):
        msg = {
            KEY_NAME: wrapped_data.get_name(),
            KEY_SIZE: wrapped_data.get_size(),
        }
        if wrapped_data.get_name() == VAL_MSG_NAME:
            msg[KEY_MSG] = wrapped_data.get_data()
            msg[KEY_QUIET] = wrapped_data.is_quiet()
        msg_binary = json.dumps(msg).encode("utf-8")
        self.send_data_bytes(stream, config, msg_binary, stop_event, send_event_queue)

    def received_handshake(self, file_desc):
        content = file_desc.read()
        if content.startswith(b"{") and content.endswith(
                b"}") and KEY_NAME.encode("utf-8") in content:
            try:
                res = json.loads(content)
                return res
            except:
                pass
        return None

    def reload_recv_speed(self, kbps):
        log.debug("Change recv speed to %skbps" % kbps)
        idx = 0  # listen thread
        self.ready_events[idx].clear()
        self.stop_events[idx].set()
        self.recv_cfg = create_config(kbps)
        self.stop_events[idx].clear()
        self.ready_events[idx].set()

    def reload_send_speed(self, kbps):
        idx = 1  # listen thread
        log.debug("Change recv send to %skbps" % kbps)
        self.ready_events[idx].clear()
        self.stop_events[idx].set()
        self.send_cfg = create_config(kbps)
        self.stop_events[idx].clear()
        self.ready_events[idx].set()

    def negotiate_speed(self):
        pass

    def request_negotiate(self, kbps):
        # data = WrappedData(VAL_MSG_NAME, NEGOT_START + f";{kbps}", quiet=True)
        # self.send_handshake(self.negot_s_stream, self.negot_cfg, data, self.stop_events[3], self.negot_event_queue)
        pass

    def response_negotiate(self, kbps):
        pass

    def send_loop(self, stream_loader, config_loader, task_queue, ready_event: threading.Event,
                  stop_event: threading.Event,
                  send_event_queue: queue.Queue):
        ready_event.set()
        while self.listening:
            try:
                ready_event.wait()
                stream = stream_loader()
                config = config_loader()
                # 使用带超时的 get 操作替代 empty 检查和 sleep
                wrapped_data = task_queue.get(timeout=0.1)
                self.send_handshake(stream, config, wrapped_data, stop_event, send_event_queue)
                if wrapped_data.get_name() != VAL_MSG_NAME:
                    send_event_queue.put(Event(Evt.SEND_FILE_START, ""))
                    self.send_data_bytes(stream, config, wrapped_data.get_data(), stop_event, send_event_queue)
            except queue.Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                # 处理其他可能的异常
                import traceback
                traceback.print_exc()

    def listen_loop(self, stream_loader, config_loader, ready_event: threading.Event, stop_event: threading.Event,
                    listen_event_queue: queue.Queue):
        ready_event.set()
        while self.listening:
            ready_event.wait()
            stream = stream_loader()
            config = config_loader()
            tmpf = self.receive_data_signal(stream, config, stop_event=stop_event)
            if not tmpf:
                continue
            handshake = self.received_handshake(tmpf)
            if isinstance(handshake, dict):
                if handshake.get(KEY_NAME) == VAL_MSG_NAME:
                    if not handshake.get(KEY_QUIET):
                        self.notify_event_queue.put(Event(Evt.NOTIFY_MSG, handshake.get(KEY_MSG)))
                    if handshake.get(KEY_MSG, "").startswith(NEGOT_START):
                        # self.reload_ todo
                        # self.data_task_queue.put(WrappedData(KEY_MSG, ))
                        pass
                else:
                    self.notify_event_queue.put(Event(Evt.NOTIFY_FILE, handshake.get(KEY_NAME)))

                    folder = os.path.join(CWD, "received")
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    file_name = handshake.get(KEY_NAME)

                    listen_event_queue.put(Event(Evt.RECV_FILE_START, handshake.get(KEY_NAME), handshake.get(KEY_SIZE)))

                    send_time = config.silence_start + config.silence_stop + handshake.get(
                        KEY_SIZE) * 8 / config.modem_bps
                    listen_event_queue.put(Event(Evt.SPEND_TIME, send_time))

                    f_path = os.path.join(folder, file_name)
                    with open(f_path, "wb+") as temp_file:
                        self.receive_data_signal(stream, config, dst=temp_file, stop_event=stop_event)

                    listen_event_queue.put(Event(Evt.RECV_FILE_FINISH, f_path))

    def stop(self):
        self.listening = False
        for event in self.stop_events:
            event.set()
        for stream in self.opened_streams:
            if stream:
                stream.close()


def test_send_msg():
    base = SoundChannelBase()
    with open("data.txt", "r", encoding="utf-8") as fd:
        content = fd.read()
    base.send_message(content)
    time.sleep(3)
    base.stop()


def test_send_file():
    base = SoundChannelBase()
    base.send_file("test.txt")
    time.sleep(3)
    base.stop()


if __name__ == "__main__":
    # test_send_msg()
    # test_send_file()
    pass
