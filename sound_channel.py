import itertools
import json
import logging
import os
import queue
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

log = logging.getLogger('__name__')


class _tmp:
    verbose = 2
    quiet = False


_config_log(_tmp())

MSG_NAME = "\\msg_str/"
FORMAT = pyaudio.paInt16
CHANNELS = 1
kb_per_s = 32
CFG = bitrates[kb_per_s]
CFG.silence_start = CFG.silence_stop = 0.15
CFG.timeout = float("inf")
CWD = os.path.dirname(os.path.abspath(__file__))


class Evt(Enum):
    SPEND_TIME = "SEND_TIME"
    SEND_FILE_START = "SEND_FILE_START"
    SEND_FINISH = "SEND_FINISH"
    RECV_FILE_START = "RECV_FILE_START"
    RECV_FILE_FINISH = "RECV_FILE_FINISH"
    NOTIFY_MSG = "NOTIFY_MSG"
    NOTIFY_FILE = "NOTIFY_FILE"
    RE_TRANSMIT = "RE_TRANSMIT"


def text_to_bytes(text: str):
    return text.encode('utf-8')


def bytes_to_text(binary_text: bytes):
    return binary_text.decode('utf-8')


def bytes_to_bits(bytes_data: bytes):
    return np.unpackbits(np.frombuffer(bytes_data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray):
    padding = (8 - len(bits) % 8) % 8
    bits = np.pad(bits, (0, padding), 'constant')
    bit_array = bits.reshape(-1, 8)
    byte_array = np.packbits(bit_array, axis=1)
    return byte_array.tobytes()


class Event:
    def __init__(self, key, value, o1=None, o2=None, o3=None):
        self.key = key
        self.value = value
        self.o1 = o1
        self.o2 = o2
        self.o3 = o3


class WrappedData:

    def __init__(self, data_name, data_content):
        self.data_name = data_name
        self.data_content = data_content
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


class SoundChannelBase:

    def __init__(self, ):
        self.task_queue = queue.Queue()

        self.interface = audio.Interface(CFG)
        self.interface.load("")
        self.notify_event_queue = queue.Queue()
        self.recv_event_queue = queue.Queue()
        self.send_event_queue = queue.Queue()
        self.rx_signal_queue = queue.Queue()
        self.listening = True
        self.r_stream = self.create_recv_stream()
        self.s_stream = self.create_send_stream()
        l_start_event = threading.Event()
        s_start_event = threading.Event()
        self.processes = []
        self.p_listen = threading.Thread(target=self.listen_loop,
                                         args=(l_start_event, self.recv_event_queue,), daemon=True)
        self.p_send = threading.Thread(target=self.send_loop,
                                       args=(s_start_event, self.send_event_queue,), daemon=True)
        self.processes.append(self.p_listen)
        self.processes.append(self.p_send)
        self.p_listen.start()
        self.p_send.start()
        l_start_event.wait()
        s_start_event.wait()

    def create_send_stream(self):
        return self.interface.player()

    def create_recv_stream(self):
        recv_stream = self.interface.recorder()
        return AsyncReader(recv_stream, recv_stream.bufsize)

    def send_message(self, msg):
        if isinstance(msg, str):
            self.queue_task_data(msg)

    def send_file(self, file_path):
        _, file_name = os.path.split(file_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            with open(file_path, 'rb') as fd:
                data = fd.read()
                self.queue_task_data(data, filename=file_name)

    def queue_task_data(self, data, filename=None):
        if filename:
            self.task_queue.put(WrappedData(filename, data))
        else:
            self.task_queue.put(WrappedData(MSG_NAME, data))

    def receive_data_signal(self, dst=None):
        reader = Reader(self.r_stream, data_type=common.loads)
        signal = itertools.chain.from_iterable(reader)

        pylab = common.Dummy()
        detector = Detector(config=CFG, pylab=pylab)
        receiver = Receiver(config=CFG, pylab=pylab)
        dst = dst or tempfile.TemporaryFile()
        try:
            log.info('Waiting for carrier tone: %.1f kHz' % (CFG.Fc / 1e3))
            signal, amplitude, freq_error = detector.run(signal)

            freq = 1 / (1.0 + freq_error)  # receiver's compensated frequency
            log.debug('Frequency correction: %.3f ppm' % ((freq - 1) * 1e6))

            gain = 1.0 / amplitude
            log.debug('Gain correction: %.3f' % gain)

            sampler = sampling.Sampler(signal,
                                       sampling.defaultInterpolator,
                                       freq=freq)
            receiver.run(sampler, gain=1.0 / amplitude, output=dst)
            return dst
        except BaseException:  # pylint: disable=broad-except
            import traceback
            traceback.print_exc()
            log.error('Decoding failed')
            return False
        finally:
            dst.flush()
            dst.seek(0)
            receiver.report()

    def send_data_bytes(self, bytes_data: bytes):
        t0 = time.time()
        send_time = CFG.silence_start + CFG.silence_stop + len(bytes_data) * 8 / (1000 * kb_per_s)
        self.send_event_queue.put(Event(Evt.SPEND_TIME, send_time))

        sender = Sender(self.s_stream, config=CFG)
        # pre-padding audio with silence (priming the audio sending queue)
        sender.write(np.zeros(int(CFG.Fs * CFG.silence_start)))
        sender.start()

        training_duration = sender.offset
        log.info('Sending %.3f seconds of training audio' % (training_duration / CFG.Fs))
        framer = Framer()
        bits = framing.encode(bytes_data, framer=framer)
        log.info('Starting modulation')
        sender.modulate(bits=bits, event_queue=self.rx_signal_queue, framer=framer)

        data_duration = sender.offset - training_duration
        log.info('Sent %.3f kB @ %.3f seconds' % (len(bytes_data) / 1e3, data_duration / CFG.Fs))

        # post-padding audio with silence
        sender.write(np.zeros(int(CFG.Fs * CFG.silence_stop)))
        t1 = time.time()
        log.info("Total Used time %.4f seconds" % (t1 - t0))

        self.send_event_queue.put(Event(Evt.SEND_FINISH, ""))
        return

    def send_handshake(self, wrapped_data: WrappedData):
        msg = {
            "name": wrapped_data.get_name(),
            "size": wrapped_data.get_size(),
        }
        if wrapped_data.get_name() == MSG_NAME:
            msg["msg"] = wrapped_data.get_data()
        msg_binary = json.dumps(msg).encode("utf-8")
        self.send_data_bytes(msg_binary)

    def received_handshake(self, file_desc):
        content = file_desc.read()
        if content.startswith(b'{') and content.endswith(
                b"}") and b'"name"' in content:
            try:
                res = json.loads(content)
                return res
            except:
                pass
        return False

    def send_loop(self, start_event: threading.Event, event_queue: queue.Queue):
        start_event.set()
        while self.listening:
            try:
                # 使用带超时的 get 操作替代 empty 检查和 sleep
                wrapped_data = self.task_queue.get(timeout=0.1)
                self.send_handshake(wrapped_data)
                if wrapped_data.get_name() != MSG_NAME:
                    event_queue.put(Event(Evt.SEND_FILE_START, ""))
                    self.send_data_bytes(wrapped_data.get_data())
            except queue.Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                # 处理其他可能的异常
                import traceback
                traceback.print_exc()

    def listen_loop(self, start_event: threading.Event, event_queue: queue.Queue):
        start_event.set()
        while self.listening:
            tmpf = self.receive_data_signal()
            if not tmpf:
                continue
            handshake = self.received_handshake(tmpf)
            if handshake:
                if handshake.get("name") == MSG_NAME:
                    self.notify_event_queue.put(Event(Evt.NOTIFY_MSG, handshake.get("msg")))
                else:
                    self.notify_event_queue.put(Event(Evt.NOTIFY_FILE, handshake.get("name")))

                    folder = os.path.join(CWD, "received")
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    file_name = handshake.get("name")

                    event_queue.put(Event(Evt.RECV_FILE_START, handshake.get("name"), handshake.get("size")))

                    send_time = CFG.silence_start + CFG.silence_stop + handshake.get("size") * 8 / (1000 * kb_per_s)
                    event_queue.put(Event(Evt.SPEND_TIME, send_time))

                    f_path = os.path.join(folder, file_name)
                    with open(f_path, 'wb+') as temp_file:
                        self.receive_data_signal(dst=temp_file)

                    event_queue.put(Event(Evt.RECV_FILE_FINISH, f_path))
        self.r_stream.close()

    def stop(self):
        self.listening = False


def test_send_msg():
    base = SoundChannelBase()
    with open("data.txt", 'r', encoding="utf-8") as fd:
        content = fd.read()
    base.send_message(content)
    base.stop()


def test_send_file():
    base = SoundChannelBase()
    base.send_file("test.txt")
    base.stop()


if __name__ == "__main__":
    test_send_msg()
    # test_send_file()
