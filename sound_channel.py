import itertools
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
import zlib

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
from amodem.framing import Framer, BitPacker
from amodem.recv import Receiver
from amodem.send import Sender
from amodem.stream import Reader
from enum import Enum

log = logging.getLogger("__name__")


class _tmp:
    verbose = 2
    quiet = False


_config_log(_tmp())

KEY_NAME = "name"
KEY_MSG = "msg"
KEY_SIZE = "size"
KEY_QUIET = "is_quiet"

VAL_MSG_NAME = "\\msg_str/"

SEPARATOR = ";;"
SPECIAL = "~!@#$%^&*()_+"
NEGOT_START = "NEGOT_START" + SPECIAL
NEGOT_END = "NEGOT_END" + SPECIAL
NEGOT_RECV = "NEGOT_RECV" + SPECIAL
NEGOT_SEND = "NEGOT_SEND" + SPECIAL
NEGOT_SNR = "NEGOT_SNR" + SPECIAL
NEGOT_RECV_STATE = "NEGOT_RECV_STATE" + SPECIAL
NEGOT_SEND_STATE = "NEGOT_SEND_STATE" + SPECIAL
NEGOT_RESEND_FRAME = "NEGOT_RESEND_FRAME" + SPECIAL

STATE_PASS = "pass"
STATE_FAIL = "fail"
STATE_CANCEL = "cancel"
TYPE_SEND = "send"
TYPE_RECV = "recv"

FORMAT = pyaudio.paInt16
CHANNELS = 1
INIT_SEND_kbps = 40
INIT_RECV_kbps = 40


class Evt(Enum):
    SEND_FILE_START = "SEND_FILE_START"
    SEND_FINISH = "SEND_FINISH"
    RECV_FILE_START = "RECV_FILE_START"
    RECV_FILE_FINISH = "RECV_FILE_FINISH"
    FILE_FAIL = "FILE_FAIL"
    FILE_CANCEL = "FILE_CANCEL"
    NOTIFY_MSG = "NOTIFY_MSG"
    NOTIFY_FILE = "NOTIFY_FILE"
    NOTIFY_NEGOT = "NOTIFY_NEGOT"


class Event:
    def __init__(self, key, value, o1=None, o2=None, o3=None):
        self.key = key
        self.value = value
        self.o1 = o1
        self.o2 = o2
        self.o3 = o3


def get_application_path():
    if getattr(sys, "frozen", False):
        # 如果是打包后的应用程序
        return os.path.dirname(sys.executable)
    else:
        # 如果是在开发环境中运行
        return os.path.dirname(os.path.abspath(__file__))


CWD = get_application_path()


def generate_rates():
    desc_to_int = {}
    int_to_desc = {}
    for key, val in bitrates.items():
        desc_to_int[f"{key}_QAM{val.Npoints}"] = key
        int_to_desc[key] = f"{key}_QAM{val.Npoints}"
    return desc_to_int, int_to_desc


RATES_DESC_TO_IDX, RATES_IDX_TO_DESC = generate_rates()


def create_config(kbps):
    config = bitrates[kbps]
    config.silence_start = config.silence_stop = 0.1
    config.timeout = float("inf")
    return config


def compress_data(bytes_data):
    return zlib.compress(bytes_data)


def decompress_temp_file(compressed_file, decompressed_file=None):
    # 创建一个新的临时文件来存储解压后的数据
    decompressed_file = decompressed_file or tempfile.TemporaryFile()
    # 将压缩文件的指针移到开始
    compressed_file.seek(0)
    # 创建一个zlib解压对象
    decompressor = zlib.decompressobj()
    try:
        # 分块读取并解压数据
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = compressed_file.read(chunk_size)
            if not chunk:
                break
            decompressed_chunk = decompressor.decompress(chunk)
            decompressed_file.write(decompressed_chunk)
        # 确保所有剩余的数据都被刷新出来
        final_chunk = decompressor.flush()
        decompressed_file.write(final_chunk)
        # 将解压文件的指针移到开始
        decompressed_file.seek(0)
        return decompressed_file
    except zlib.error as e:
        log.info(f"Uncompress fail: {e}")
        decompressed_file.close()
        return None


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
        self.cut_eof = True
        self.recv_cfg = create_config(INIT_SEND_kbps)
        self.send_cfg = create_config(INIT_RECV_kbps)
        self.negot_cfg = Configuration(Fs=32e3, Npoints=16, frequencies=self.send_cfg.negotiate_frequencies)
        self.negot_cfg.timeout = float("inf")

        self.data_interface = self.create_interface(self.send_cfg)
        self.negot_interface = self.create_interface(self.negot_cfg)

        self.opened_streams = []
        self.r_stream = self.create_recv_stream(self.data_interface)
        self.s_stream = self.create_send_stream(self.data_interface)
        self.negot_r_stream = self.create_recv_stream(self.negot_interface)
        self.negot_s_stream = self.create_send_stream(self.negot_interface)
        self.opened_streams.append(self.r_stream)
        self.opened_streams.append(self.s_stream)
        self.opened_streams.append(self.negot_r_stream)
        self.opened_streams.append(self.negot_s_stream)

        self.listening = True

        self.notify_event_queue = queue.Queue()  # notify to ui
        self.recv_event_queue = queue.Queue()  # recv event to ui
        self.send_event_queue = queue.Queue()  # send event to ui
        self.negot_event_queue = queue.Queue()  # send negotiate event to ui
        self.frame_resend_queue = queue.Queue()  # send negotiate event to ui

        self.data_task_queue = queue.Queue()
        self.negot_task_queue = queue.Queue()

        self.ready_events = [threading.Event() for _ in range(4)]
        self.stop_events = [threading.Event() for _ in range(4)]
        self.running_threads = []
        self.p_listen = threading.Thread(
            target=self.listen_loop,
            args=(self.load_r_stream, self.load_recv_cfg, self.ready_events[0], self.stop_events[0],
                  self.recv_event_queue,))
        self.p_send = threading.Thread(
            target=self.send_loop,
            args=(self.load_s_stream, self.load_send_cfg, self.data_task_queue, self.ready_events[1],
                  self.stop_events[1], self.send_event_queue,))
        self.negot_listen = threading.Thread(
            target=self.listen_loop,
            args=(self.load_negot_r_stream, self.load_negot_cfg, self.ready_events[2], self.stop_events[2],
                  self.negot_event_queue,))
        self.negot_send = threading.Thread(
            target=self.send_loop,
            args=(self.load_negot_s_stream, self.load_negot_cfg, self.negot_task_queue, self.ready_events[3],
                  self.stop_events[3], self.negot_event_queue,))
        self.running_threads.append(self.p_listen)
        self.running_threads.append(self.p_send)
        self.running_threads.append(self.negot_listen)
        self.running_threads.append(self.negot_send)
        for th in self.running_threads:
            th.daemon = True
            th.start()
        for ready_evt in self.ready_events:
            ready_evt.wait()
        data = WrappedData(VAL_MSG_NAME, NEGOT_START, quiet=True)
        self.negot_task_queue.put(data)

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
        compressed_file = self.receive_data_signal_compressed(stream, config, stop_event=stop_event)
        if compressed_file is None:
            return
        decompressed = decompress_temp_file(compressed_file, decompressed_file=dst)
        return decompressed

    def receive_data_signal_compressed(self, stream, config, stop_event: threading.Event = None):
        reader = Reader(stream, data_type=common.loads)
        signal = itertools.chain.from_iterable(reader)

        pylab = common.Dummy()
        detector = Detector(config=config, pylab=pylab)
        receiver = Receiver(config=config, pylab=pylab)
        dst = tempfile.TemporaryFile()
        try:
            log.info("Waiting for carrier tone: %.1f kHz" % (config.Fc / 1e3))
            signal, amplitude, freq_error = detector.run(signal, stop_event=stop_event)

            freq = 1 / (1.0 + freq_error)  # receiver's compensated frequency
            log.debug("Frequency correction: %.3f ppm" % ((freq - 1) * 1e6))

            gain = 1.0 / amplitude
            log.debug("Gain correction: %.3f" % gain)

            sampler = sampling.Sampler(signal,
                                       sampling.defaultInterpolator,
                                       freq=freq)
            receiver.run(sampler, gain=1.0 / amplitude, output=dst, cut_eof=self.cut_eof)
            return dst
        except BaseException:  # pylint: disable=broad-except
            traceback.print_exc()
            log.error("Decoding failed")
            return None
        finally:
            dst.flush()
            dst.seek(0)
            receiver.report()

    def send_data_bytes(self, stream, config, bytes_data, stop_event: threading.Event, send_event_queue: queue.Queue):
        t0 = time.time()

        sender = Sender(stream, config=config)
        # pre-padding audio with silence (priming the audio sending queue)
        sender.write(np.zeros(int(config.Fs * config.silence_start)))
        sender.start()

        training_duration = sender.offset
        log.info("Sending %.3f seconds of training audio" % (training_duration / config.Fs))
        log.info("Starting compress data")
        compressed = compress_data(bytes_data)
        framer = Framer()
        bits = framing.encode(compressed, framer=framer, cut_eof=self.cut_eof)
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

    def pause_loop_thread(self, idx):
        self.ready_events[idx].clear()
        self.stop_events[idx].set()
        while self.stop_events[idx].is_set():
            time.sleep(0.05)

    def resume_loop_thread(self, idx):
        self.ready_events[idx].set()

    def cancel_data_sending(self):
        log.info("Cancel data sending")
        idx = 1  # send thread id
        self.pause_loop_thread(idx)
        self.resume_loop_thread(idx)

    def cancel_data_receiving(self):
        log.info("Cancel data receiving")
        idx = 0  # send thread id
        self.pause_loop_thread(idx)
        self.resume_loop_thread(idx)

    def negot_remote_send_speed(self, kbps):
        data = WrappedData(VAL_MSG_NAME, NEGOT_SEND + f"{SEPARATOR}{kbps}", quiet=True)
        self.negot_task_queue.put(data)
        while not self.negot_task_queue.empty():
            time.sleep(0.05)

    def reload_recv_speed(self, kbps):
        if kbps == self.recv_cfg.modem_bps // 1000:
            return False
        log.info("Change recv speed to %s kbps" % kbps)
        idx = 0  # listen thread id
        self.pause_loop_thread(idx)
        self.recv_cfg = create_config(kbps)
        self.resume_loop_thread(idx)
        return True

    def negot_remote_recv_speed(self, kbps):
        data = WrappedData(VAL_MSG_NAME, NEGOT_RECV + f"{SEPARATOR}{kbps}", quiet=True)
        self.negot_task_queue.put(data)
        while not self.negot_task_queue.empty():
            time.sleep(0.05)

    def reload_send_speed(self, kbps):
        if kbps == self.send_cfg.modem_bps // 1000:
            return False

        idx = 1  # send thread id
        log.info("Change send speed to %s kbps" % kbps)
        self.pause_loop_thread(idx)
        self.send_cfg = create_config(kbps)
        self.resume_loop_thread(idx)
        return True

    def negot_speed(self):
        pass

    def request_negot(self, kbps):
        pass

    def response_negot(self, kbps):
        pass

    def negot_remote_cancel_recv(self):
        self.negot_task_queue.put(
            WrappedData(VAL_MSG_NAME, NEGOT_RECV_STATE + f"{SEPARATOR}{STATE_CANCEL}", quiet=True))

    def negot_remote_cancel_send(self):
        self.negot_task_queue.put(
            WrappedData(VAL_MSG_NAME, NEGOT_SEND_STATE + f"{SEPARATOR}{STATE_CANCEL}", quiet=True))

    def negot_local_recv_fail(self):
        self.negot_task_queue.put(
            WrappedData(VAL_MSG_NAME, NEGOT_SEND_STATE + f"{SEPARATOR}{STATE_FAIL}", quiet=True))

    def check_negot_msg(self, msg):
        if msg.startswith(NEGOT_SEND):
            try:
                kbps = int(msg.split(SEPARATOR)[-1])
                self.notify_event_queue.put(Event(Evt.NOTIFY_NEGOT, TYPE_SEND, o1=kbps))
                if self.reload_send_speed(kbps):
                    self.notify_event_queue.put(
                        Event(Evt.NOTIFY_MSG, f"Negotiate your sent speed as {kbps} kbps"))
            except:
                traceback.print_exc()
        elif msg.startswith(NEGOT_RECV):
            try:
                kbps = int(msg.split(SEPARATOR)[-1])
                self.notify_event_queue.put(Event(Evt.NOTIFY_NEGOT, TYPE_RECV, o1=kbps))
                if self.reload_recv_speed(kbps):
                    self.notify_event_queue.put(
                        Event(Evt.NOTIFY_MSG, f"Negotiate your recv speed as {kbps} kbps"))
            except:
                traceback.print_exc()
        elif msg.startswith(NEGOT_RECV_STATE):
            try:
                state = msg.split(SEPARATOR)[-1]
                idx = 0  # listen thread id
                if state == STATE_CANCEL:
                    self.pause_loop_thread(idx)
                    self.resume_loop_thread(idx)
                    self.send_event_queue.put(Event(Evt.FILE_CANCEL, ""))
            except:
                traceback.print_exc()
        elif msg.startswith(NEGOT_SEND_STATE):
            try:
                state = msg.split(SEPARATOR)[-1]
                idx = 1  # send thread id
                if state == STATE_FAIL:
                    self.pause_loop_thread(idx)
                    self.send_event_queue.put(Event(Evt.FILE_FAIL, ""))
                    self.resume_loop_thread(idx)
                elif state == STATE_CANCEL:
                    self.pause_loop_thread(idx)
                    self.resume_loop_thread(idx)
                    self.send_event_queue.put(Event(Evt.FILE_CANCEL, ""))
            except:
                traceback.print_exc()

    def send_loop(self, stream_loader, config_loader, task_queue, ready_event: threading.Event,
                  stop_event: threading.Event,
                  send_event_queue: queue.Queue):
        ready_event.set()
        while self.listening:
            try:
                stop_event.clear()
                ready_event.wait()
                stream = stream_loader()
                config = config_loader()
                # 使用带超时的 get 操作替代 empty 检查和 sleep
                wrapped_data = task_queue.get(timeout=0.1)
                self.send_handshake(stream, config, wrapped_data, stop_event, send_event_queue)
                if wrapped_data.get_name() != VAL_MSG_NAME:
                    bytes_data = wrapped_data.get_data()
                    send_time = config.silence_start + config.silence_stop + len(bytes_data) * 8 / config.modem_bps
                    send_event_queue.put(
                        Event(Evt.SEND_FILE_START, wrapped_data.get_name(), wrapped_data.get_size(), send_time))
                    self.send_data_bytes(stream, config, bytes_data, stop_event, send_event_queue)
            except queue.Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                # 处理其他可能的异常
                traceback.print_exc()

    def listen_loop(self, stream_loader, config_loader, ready_event: threading.Event, stop_event: threading.Event,
                    listen_event_queue: queue.Queue):
        ready_event.set()
        while self.listening:
            stop_event.clear()
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
                    msg = handshake.get(KEY_MSG, "")
                    self.check_negot_msg(msg)
                else:
                    self.notify_event_queue.put(Event(Evt.NOTIFY_FILE, handshake.get(KEY_NAME)))

                    folder = os.path.join(CWD, "received")
                    if not os.path.exists(folder):
                        os.mkdir(folder)

                    file_name = handshake.get(KEY_NAME)
                    send_time = config.silence_start + config.silence_stop + handshake.get(
                        KEY_SIZE) * 8 / config.modem_bps
                    listen_event_queue.put(
                        Event(Evt.RECV_FILE_START, handshake.get(KEY_NAME), handshake.get(KEY_SIZE), send_time))

                    f_path = os.path.join(folder, file_name)
                    with open(f_path, "wb+") as temp_file:
                        recv_ok = self.receive_data_signal(stream, config, dst=temp_file, stop_event=stop_event)
                    if recv_ok:
                        listen_event_queue.put(Event(Evt.RECV_FILE_FINISH, f_path))
                    else:
                        listen_event_queue.put(Event(Evt.FILE_FAIL, f_path))
                        if stream is self.r_stream and not stop_event.is_set():
                            self.negot_local_recv_fail()

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
