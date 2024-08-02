import itertools
import json
import logging
import math
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
import zlib
from collections import defaultdict
from enum import Enum

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

log = logging.getLogger("__name__")


class _tmp:
    verbose = 3
    quiet = False


_config_log(_tmp())

KEY_NAME = "name"
KEY_MSG = "msg"
KEY_SIZE = "size"
KEY_QUIET = "is_quiet"

VAL_MSG_NAME = "<msg_str>"

SEP = ";;"
MULTI_FILE_SEP = ".part."

NEGOT_TAG = "~!@#$%^&*()_+"
NEGOT_START = "NEGOT_START" + NEGOT_TAG
NEGOT_RECV_SPEED = "NEGOT_RECV_SPEED" + NEGOT_TAG
NEGOT_SEND_SPEED = "NEGOT_SEND_SPEED" + NEGOT_TAG
NEGOT_RECV_MULTIPART_FILE = "NEGOT_RECV_MULTIPART_FILE" + NEGOT_TAG
NEGOT_SEND_MULTIPART_FILE = "NEGOT_SEND_MULTIPART_FILE" + NEGOT_TAG
NEGOT_RECV_STATE = "NEGOT_RECV_STATE" + NEGOT_TAG
NEGOT_SEND_STATE = "NEGOT_SEND_STATE" + NEGOT_TAG
NEGOT_RESEND_FRAME = "NEGOT_RESEND_FRAME" + NEGOT_TAG
NEGOT_RESEND_WHOLE = "NEGOT_RESEND_WHOLE" + NEGOT_TAG

STATE_PASS = "pass"
STATE_FAIL = "fail"
STATE_CANCEL = "cancel"
TYPE_SEND = "send"
TYPE_RECV = "recv"
UNKNOWN = "unknown"

FORMAT = pyaudio.paInt16
CHANNELS = 1
INIT_SEND_kbps = 40
INIT_RECV_kbps = 40
FILE_UNIT_KB = 512
MULTI_FILE_SIZE_BYTES = 1024 * FILE_UNIT_KB
NEGOT_CFG = Configuration(Fs=32e3, Npoints=16, frequencies=Configuration.negotiate_frequencies)


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
RECEIVE_FOLDER = os.path.join(CWD, "received")
if not os.path.exists(RECEIVE_FOLDER):
    os.mkdir(RECEIVE_FOLDER)


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
    config.silence_start = 0.1
    config.silence_stop = 0.4
    config.timeout = float("inf")
    return config


def cal_send_time(config: Configuration, data_bytes_size):
    data_per_frame = (framing.Framer.block_size - 4)  # 4 bytes frame_id
    silence_time = config.silence_start + (250 + 50 + 200 + 50) / 1000 + config.silence_stop
    data_time = (math.ceil(data_bytes_size / data_per_frame) * framing.Framer.chunk_size) * 8 / config.modem_bps
    return silence_time + data_time


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


def get_multipart_tail(index):
    return f"{MULTI_FILE_SEP}{index + 1:0>3}"


def gen_multipart_tails(multipart_size):
    return [get_multipart_tail(_) for _ in range(multipart_size)]


def check_crc32(filename, file_crc32):
    file_path = os.path.join(RECEIVE_FOLDER, filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, 'rb') as fd:
            tmp_crc32 = framing._checksum_func(fd.read())
        return tmp_crc32 == file_crc32
    return False


def check_local_existing_file(filename, file_crc32, part_name_list, part_crc32_list):
    needed_part_indexes = []
    if check_crc32(filename, file_crc32):
        pass  # file exists, no need transfer
    else:
        for i, part_name in enumerate(part_name_list):
            if not check_crc32(part_name, part_crc32_list[i]):
                needed_part_indexes.append(i)
    return needed_part_indexes


def find_increasing_sequence(numbers, n=4):
    if len(numbers) < n:
        return None
    count = 1
    start = numbers[0]
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            count += 1
            if count == n:
                return start
        else:
            count = 1
            start = numbers[i]
    return None


def merge_files(file_list, output_file, delete_on_finish=True):
    with open(output_file, 'w+b') as outfile:
        for file_path in file_list:
            with open(file_path, 'rb') as infile:
                outfile.write(infile.read())
            if delete_on_finish:
                os.remove(file_path)


class DualQueue:
    def __init__(self):
        self.high = queue.Queue()
        self.low = queue.Queue()


class TaggedQueue:
    def __init__(self):
        self.queues = defaultdict(DualQueue)
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.events = defaultdict(threading.Event)

    def _put(self, item, tag, high_prior=False):
        with self.lock:
            if high_prior:
                self.queues[tag].high.put(item)
            else:
                self.queues[tag].low.put(item)
            self.events[tag].set()
            self.condition.notify_all()

    def _get(self, tag, block=True, timeout=None):
        with self.lock:
            if not block:
                return self._get_nowait(tag)

            end_time = None if timeout is None else time.time() + timeout

            while True:
                try:
                    return self._get_nowait(tag)
                except queue.Empty:
                    if end_time is not None and time.time() >= end_time:
                        raise queue.Empty(f"Timeout waiting for item with tag {tag}")

                    self.events[tag].clear()
                    self.condition.wait(timeout)
                    if not self.events[tag].is_set():
                        continue

    def _get_nowait(self, tag):
        try:
            return self.queues[tag].high.get_nowait()
        except queue.Empty:
            try:
                return self.queues[tag].low.get_nowait()
            except queue.Empty:
                raise queue.Empty(f"No item available for tag {tag}")

    def _empty(self, tag):
        return self.queues[tag].high.empty() and self.queues[tag].low.empty()


class QueueWithTag:
    def __init__(self, tagged_queue, tag):
        super().__init__()
        self.tagged_queue: TaggedQueue = tagged_queue
        self.tag = tag

    def put(self, item, high_prior=False):
        self.tagged_queue._put(item, tag=self.tag, high_prior=high_prior)

    def get(self, block=True, timeout=None):
        return self.tagged_queue._get(self.tag, block=block, timeout=timeout)

    def empty(self):
        return self.tagged_queue._empty(self.tag)


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


class ResendFrameBitsIterator:
    def __init__(self, data_with_fid, filename=None):
        self.data_with_fid = data_with_fid
        self.filename = UNKNOWN if filename is None else filename
        self.last_fid = -1
        self.id_to_frame_bits = {}
        self.resend_queue = queue.Queue()
        self.current_bits = iter([])  # 用于存储当前正在处理的位序列
        self.can_iter_resend = False

    def __iter__(self):
        return self

    def get_resend_bits(self):
        while not self.resend_queue.empty():
            fid = self.resend_queue.get()
            if fid in self.id_to_frame_bits:
                self.last_fid = fid
                return iter(self.id_to_frame_bits[fid])
        return None  # 表示重发队列已经空了

    def get_next_bits(self):
        if self.can_iter_resend:
            resend_bits = self.get_resend_bits()
            if resend_bits is not None:
                return resend_bits
            self.can_iter_resend = False  # 重发队列为空，切换回正常模式

        # 如果找不到请求重发的帧，继续获取下一个
        try:
            bits = next(self.data_with_fid)
            fid = next(self.data_with_fid)
            if fid not in self.id_to_frame_bits:
                self.id_to_frame_bits[fid] = []
            self.id_to_frame_bits[fid].extend(bits)
            if self.last_fid != fid:
                # make sure fid changes
                self.can_iter_resend = True
                self.resend_queue.put(fid)
                return self.get_next_bits()  # 递归调用以处理新加入的重发帧
            else:
                self.last_fid = fid
                return iter(bits)
        except StopIteration:
            self.can_iter_resend = True
            if self.resend_queue.empty():
                raise StopIteration
            return self.get_next_bits()

    def __next__(self):
        try:
            return next(self.current_bits)
        except StopIteration:
            self.current_bits = self.get_next_bits()
            return next(self.current_bits)

    def arrange_resend_frame(self, frame_id):
        if frame_id in self.id_to_frame_bits:
            self.resend_queue.put(frame_id)

    def arrange_resend_frames_from_id(self, frame_id):
        for fid in range(frame_id, max(self.id_to_frame_bits.keys()) + 1):
            if fid in self.id_to_frame_bits:
                self.resend_queue.put(fid)


class Frame:
    def __init__(self, frame_id, data):
        self.frame_id = frame_id
        self.data = data


class FrameManager:
    def __init__(self):
        self.frames_list = []  # 用于保持有序的frame id
        self.frames_dict = {}  # 用于快速查找frame内容

    def is_continuous(self):
        return all(y - x == 1 for x, y in zip(self.frames_list, self.frames_list[1:]))

    def get_missing_frames(self):
        if not self.frames_list:
            return []  # 如果列表为空，返回空列表

        start = self.frames_list[0]
        end = self.frames_list[-1]
        all_frames = set(range(start, end + 1))
        existing_frames = set(self.frames_list)

        return sorted(all_frames - existing_frames)

    @property
    def missing_frame_ids(self):
        return set(self.get_missing_frames())

    def insert_frame(self, frame):
        # 检查frame是否已经存在
        if frame.frame_id in self.frames_dict:
            log.warning(f"Frame with id {frame.frame_id} already exists.")
            return
        # 找到应插入的位置
        index = self.find_insert_position(frame.frame_id)
        # 插入frame id到有序列表中
        self.frames_list.insert(index, frame.frame_id)
        # 插入frame到字典中
        self.frames_dict[frame.frame_id] = frame
        return index

    def find_insert_position(self, frame_id):
        # 二分查找找到插入位置
        low, high = 0, len(self.frames_list)
        while low < high:
            mid = (low + high) // 2
            if self.frames_list[mid] < frame_id:
                low = mid + 1
            else:
                high = mid
        return low

    def get_frame(self, frame_id):
        return self.frames_dict.get(frame_id)

    def remove_frame(self, frame_id):
        if frame_id in self.frames_dict:
            self.frames_list.remove(frame_id)
            del self.frames_dict[frame_id]
        else:
            log.warning(f"Frame with id {frame_id} does not exist.")


class FrameWriter(FrameManager):
    def __init__(self, filename, dst, modem_bps, negot_resend_frame):
        super().__init__()
        self.filename = UNKNOWN if filename is None else filename
        self.dst = dst
        self.negot_resend_frame = negot_resend_frame

        self.paused_error_frame_id = None
        self.error_cnt_after_paused = 0

        frame_bytes_size = framing.Framer.chunk_size
        self.frame_per_second = (modem_bps / 8) // frame_bytes_size
        self.latent_time_secs = cal_send_time(NEGOT_CFG, 1) + 1
        self.expect_frame_cnt = math.ceil(MULTI_FILE_SIZE_BYTES / frame_bytes_size)
        self.error_limit = int(self.frame_per_second * self.latent_time_secs)
        self.num_frames_per_check = max(1, self.frame_per_second)
        self.num_sequential_missing = max(2, int(round(self.num_frames_per_check / 4)))
        self.last_frame_id = -1

    def get_dst(self):
        return self.dst

    def validation(self):
        if not self.is_continuous():
            log.warning(f"Validation fail, missing frames: {self.get_missing_frames()}")
            return False
        return True

    def write(self, data, frame_id):
        if data == b'':
            if abs(frame_id - self.last_frame_id) > self.expect_frame_cnt:
                frame_id = self.last_frame_id + 1  # frame_id correction
            if self.paused_error_frame_id is not None:
                self.error_cnt_after_paused += 1
            if len(self.missing_frame_ids) > self.error_limit or self.error_cnt_after_paused > self.error_limit:
                raise Exception(f"Too many error frames > {self.error_limit}")
        else:
            if frame_id in self.missing_frame_ids:
                log.info(f"Retry receive frame success for frame_id={frame_id}")
                if self.paused_error_frame_id == frame_id:
                    self.paused_error_frame_id = None
                    self.error_cnt_after_paused = 0
            frame = Frame(frame_id, data)
            self.insert_frame(frame)
        # request resend missing frames
        if frame_id % self.num_frames_per_check == 0:
            missing_frame_ids = self.get_missing_frames()
            if missing_frame_ids:
                index = find_increasing_sequence(missing_frame_ids, n=self.num_sequential_missing)
                arrange = []
                for fid in missing_frame_ids:
                    if fid != index:
                        arrange.append(fid)
                    else:
                        arrange.append(index)
                        arrange.append(-1)
                        self.paused_error_frame_id = index
                        break
                self.negot_resend_frame(self.filename, arrange)
        if frame_id > self.last_frame_id:
            self.last_frame_id = frame_id

    def flush(self):
        self.dst.write(self.read())
        self.dst.flush()

    def read(self):
        data = b''
        for frame_id in self.frames_list:
            data += self.frames_dict[frame_id].data
        return data

    def seek(self, index):
        self.dst.seek(index)

    def close(self):
        self.dst.close()


class SoundChannelBase:

    def __init__(self, ):
        self.cut_eof = True
        self.use_frame_id = True
        self.rfbi_bits: ResendFrameBitsIterator = None
        self.filename_to_path = {}
        self.merge_files_record = {}

        self.recv_cfg = create_config(INIT_SEND_kbps)
        self.send_cfg = create_config(INIT_RECV_kbps)
        self.negot_cfg = NEGOT_CFG
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

        self.multipart_file_negot_queue = queue.Queue()  # negotiate multipart file parts to be sent
        self.filename_data_dict = {}  # temp save for resend whole

        self.sending_task_queue = TaggedQueue()
        self.data_task_queue = QueueWithTag(self.sending_task_queue, "data_task_queue")
        self.negot_task_queue = QueueWithTag(self.sending_task_queue, "negot_task_queue")

        self.sending_lock = threading.Lock()
        self.sending_data_name = None
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
        self.negot_message(NEGOT_START, "")

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

    def queue_send_data(self, task_queue, data, filename=None):
        if filename:
            self.filename_data_dict[filename] = data
            task_queue.put(WrappedData(filename, data))
        else:
            task_queue.put(WrappedData(VAL_MSG_NAME, data), high_prior=True)

    def send_message(self, msg):
        if isinstance(msg, str):
            self.queue_send_data(self.data_task_queue, msg)

    def send_file(self, file_path):
        self.filename_data_dict.clear()
        filename = os.path.basename(file_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            self.filename_to_path[filename] = file_path
            with open(file_path, "rb") as fd:
                data = fd.read()
            file_size = len(data)
            chunk_size = MULTI_FILE_SIZE_BYTES
            chunk_list = []
            for i in range(0, file_size, chunk_size):
                chunk = data[i:i + chunk_size]
                chunk_list.append(chunk)
            if len(chunk_list) == 1:
                self.queue_send_data(self.data_task_queue, data, filename=file_path)
            else:
                tail_list = gen_multipart_tails(len(chunk_list))
                file_crc32 = framing._checksum_func(data)
                part_crc32_list = [framing._checksum_func(_) for _ in chunk_list]
                self.negot_send_multipart_file(filename, len(chunk_list), file_crc32, part_crc32_list)
                for i, chunk in enumerate(chunk_list):
                    part_name = filename + tail_list[i]
                    self.multipart_file_negot_queue.put((file_path, file_size, i, part_name, chunk))

    def receive_data_signal(self, stream, config, dst=None, filename=None, stop_event: threading.Event = None):
        temp_dst = tempfile.TemporaryFile()
        if self.use_frame_id:
            fw = FrameWriter(filename, temp_dst, config.modem_bps, self.negot_resend_frame)
            ret_fw = self.receive_data_signal_compressed(stream, config, dst=fw, stop_event=stop_event)
            if ret_fw:
                fw.validation()
                fw.flush()
                compressed_file = fw.get_dst()
            else:
                compressed_file = None
        else:
            compressed_file = self.receive_data_signal_compressed(stream, config, dst=temp_dst, stop_event=stop_event)
        if compressed_file is None:
            return
        dst = dst or tempfile.TemporaryFile()
        decompressed = decompress_temp_file(compressed_file, decompressed_file=dst)
        return decompressed

    def receive_data_signal_compressed(self, stream, config, dst=None, stop_event: threading.Event = None):
        reader = Reader(stream, data_type=common.loads)
        signal = itertools.chain.from_iterable(reader)

        pylab = common.Dummy()
        detector = Detector(config=config, pylab=pylab)
        receiver = Receiver(config=config, pylab=pylab)
        dst = dst or tempfile.TemporaryFile()
        try:
            log.info("Waiting for carrier tone: %.1f kHz" % (config.Fc / 1e3))
            signal, amplitude, freq_error = detector.run(signal, stop_event=stop_event)

            freq = 1 / (1.0 + freq_error)  # receiver's compensated frequency
            log.debug("Frequency correction: %.3f ppm" % ((freq - 1) * 1e6))

            gain = 1.0 / amplitude
            log.debug("Gain correction: %.3f" % gain)

            sampler = sampling.Sampler(signal, sampling.defaultInterpolator, freq=freq)
            if not isinstance(dst, FrameWriter):
                receiver.run(sampler, gain=1.0 / amplitude, output=dst, cut_eof=self.cut_eof, raise_err=False)
            else:
                receiver.run(sampler, gain=1.0 / amplitude, output=dst, cut_eof=self.cut_eof, raise_err=False,
                             use_fid=True)
            return dst
        except BaseException:  # pylint: disable=broad-except
            traceback.print_exc()
            log.error("Decoding failed")
            return None
        finally:
            dst.flush()
            dst.seek(0)
            receiver.report()

    def send_data_bytes(self, stream, config, bytes_data, stop_event: threading.Event, send_event_queue: queue.Queue,
                        filename=None):
        t0 = time.time()

        sender = Sender(stream, config=config)
        # pre-padding audio with silence (priming the audio sending queue)
        sender.write(np.zeros(int(config.Fs * config.silence_start)))
        sender.start()

        training_duration = sender.offset
        log.info("Sending %.3f seconds of training audio" % (training_duration / config.Fs))
        log.info("Starting compress data")
        compressed = compress_data(bytes_data)

        log.info("Starting modulation")
        framer = Framer()
        if self.use_frame_id:
            bits_with_fid = framing.encode(compressed, framer=framer, cut_eof=self.cut_eof, use_fid=self.use_frame_id)
            self.rfbi_bits = ResendFrameBitsIterator(bits_with_fid, filename)
            sender.modulate(bits=self.rfbi_bits, stop_event=stop_event)
            self.rfbi_bits = None
        else:
            bits = framing.encode(compressed, framer=framer, cut_eof=self.cut_eof, use_fid=self.use_frame_id)
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

    def reload_recv_speed(self, kbps):
        if kbps == self.recv_cfg.modem_bps // 1000:
            return False
        log.info("Change recv speed to %s kbps" % kbps)
        idx = 0  # listen thread id
        self.pause_loop_thread(idx)
        self.recv_cfg = create_config(kbps)
        self.resume_loop_thread(idx)
        return True

    def reload_send_speed(self, kbps):
        if kbps == self.send_cfg.modem_bps // 1000:
            return False

        idx = 1  # send thread id
        log.info("Change send speed to %s kbps" % kbps)
        self.pause_loop_thread(idx)
        self.send_cfg = create_config(kbps)
        self.resume_loop_thread(idx)
        return True

    def negot_message(self, negot_type, negot_info):
        msg = WrappedData(VAL_MSG_NAME, negot_type + SEP + negot_info, quiet=True)
        self.negot_task_queue.put(msg)

    def negot_remote_recv_speed(self, kbps):
        self.negot_message(NEGOT_RECV_SPEED, f"{kbps}")

    def negot_remote_send_speed(self, kbps):
        self.negot_message(NEGOT_SEND_SPEED, f"{kbps}")

    def negot_remote_cancel_recv(self, filename):
        self.negot_message(NEGOT_RECV_STATE, f"{STATE_CANCEL}{SEP}{filename}")

    def negot_remote_cancel_send(self, filename):
        self.negot_message(NEGOT_SEND_STATE, f"{STATE_CANCEL}{SEP}{filename}")

    def negot_local_recv_fail(self, filename):
        self.negot_message(NEGOT_SEND_STATE, f"{STATE_FAIL}{SEP}{filename}")

    def negot_resend_frame(self, filename, frame_id_list):
        filename = UNKNOWN if filename is None else filename
        frame_ids_str = ','.join([str(_) for _ in frame_id_list])
        self.negot_message(NEGOT_RESEND_FRAME, f"{filename}{SEP}{frame_ids_str}")

    def negot_resend_whole(self, filename):
        self.negot_message(NEGOT_RESEND_WHOLE, f"{filename}")

    def negot_send_multipart_file(self, filename, multipart_size, file_crc32, part_crc32_list):
        part_crc32_str = ','.join([str(_) for _ in part_crc32_list])
        info = f"{filename}{SEP}{multipart_size}{SEP}{file_crc32}{SEP}{part_crc32_str}"
        self.negot_message(NEGOT_SEND_MULTIPART_FILE, info)

    def negot_recv_multipart_file(self, filename, needed_indexes):
        needed_indexes_str = ','.join([str(_) for _ in needed_indexes])
        info = f"{filename}{SEP}{needed_indexes_str}"
        self.negot_message(NEGOT_RECV_MULTIPART_FILE, info)

    def check_negot_msg(self, msg):
        if NEGOT_TAG in msg:
            log.debug("Negotiate msg: %s" % msg)
            print("Negotiate msg: %s" % msg)

        if msg.startswith(NEGOT_SEND_SPEED):
            try:
                items = msg.split(SEP)
                kbps = int(items[-1])
                self.notify_event_queue.put(Event(Evt.NOTIFY_NEGOT, TYPE_SEND, o1=kbps))
                if self.reload_send_speed(kbps):
                    self.notify_event_queue.put(
                        Event(Evt.NOTIFY_MSG, f"Negotiate your sent speed as {kbps} kbps"))
            except:
                traceback.print_exc()
        elif msg.startswith(NEGOT_RECV_SPEED):
            try:
                items = msg.split(SEP)
                kbps = int(items[-1])
                self.notify_event_queue.put(Event(Evt.NOTIFY_NEGOT, TYPE_RECV, o1=kbps))
                if self.reload_recv_speed(kbps):
                    self.notify_event_queue.put(
                        Event(Evt.NOTIFY_MSG, f"Negotiate your recv speed as {kbps} kbps"))
            except:
                traceback.print_exc()
        elif msg.startswith(NEGOT_RECV_STATE):
            try:
                items = msg.split(SEP)
                state = items[-1]
                idx = 0  # listen thread id
                if state == STATE_CANCEL:
                    self.pause_loop_thread(idx)
                    self.resume_loop_thread(idx)
                    self.send_event_queue.put(Event(Evt.FILE_CANCEL, ""))
            except:
                traceback.print_exc()
        elif msg.startswith(NEGOT_SEND_STATE):
            try:
                items = msg.split(SEP)
                state = items[1]
                filename = items[2]
                idx = 1  # send thread id
                if filename == self.sending_data_name:
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
        elif msg.startswith(NEGOT_RESEND_FRAME):
            try:
                items = msg.split(SEP)
                filename = items[1]
                frame_ids_str = items[-1]
                frame_ids = [int(_) for _ in frame_ids_str.split(",") if _]
                index = None
                if frame_ids and frame_ids[-1] == -1:
                    index = frame_ids[-2]
                if self.rfbi_bits is not None and self.rfbi_bits.filename == filename:
                    for frame_id in frame_ids:
                        if frame_id != index:
                            self.rfbi_bits.arrange_resend_frame(frame_id)
                        else:
                            self.rfbi_bits.arrange_resend_frames_from_id(frame_id)
                            break
            except:
                traceback.print_exc()
        elif msg.startswith(NEGOT_RESEND_WHOLE):
            try:
                items = msg.split(SEP)
                filename = items[-1]
                data = self.filename_data_dict.get(filename)
                if data is not None:
                    self.queue_send_data(self.data_task_queue, data, filename=filename)
            except:
                traceback.print_exc()
        elif msg.startswith(NEGOT_SEND_MULTIPART_FILE):
            try:
                items = msg.split(SEP)
                filename = items[1]
                multipart_size = int(items[2])
                file_crc32 = int(items[3])
                part_crc32_list = [int(_) for _ in items[4].split(",") if _]
                tails = gen_multipart_tails(multipart_size)
                part_name_list = [f"{filename}{_}" for _ in tails]
                needed_indexes = check_local_existing_file(filename, file_crc32, part_name_list, part_crc32_list)
                if needed_indexes:
                    self.merge_files_record[filename] = {_: True for _ in part_name_list}
                    for idx in needed_indexes:
                        part_name = part_name_list[idx]
                        self.merge_files_record[filename][part_name] = False
                self.negot_recv_multipart_file(filename, needed_indexes)
            except:
                traceback.print_exc()
        elif msg.startswith(NEGOT_RECV_MULTIPART_FILE):
            try:
                items = msg.split(SEP)
                filename = items[1]
                needed_indexes = [int(_) for _ in items[2].split(",") if _]
                checked = False
                while not self.multipart_file_negot_queue.empty():
                    (file_path, file_size, i, part_name, chunk) = self.multipart_file_negot_queue.get()
                    if i in needed_indexes and part_name == filename + get_multipart_tail(i):
                        self.queue_send_data(self.data_task_queue, chunk, filename=part_name)
                    if not checked and len(needed_indexes) == 0:
                        checked = True
                        self.notify_event_queue.put(Event(Evt.NOTIFY_FILE, file_path, o1=file_size))
                        self.send_event_queue.put(Event(Evt.SEND_FILE_START, file_path, file_size, 0.1))
                        self.send_event_queue.put(Event(Evt.SEND_FINISH, ""))
            except:
                traceback.print_exc()

    def send_loop(self, stream_loader, config_loader, task_queue: QueueWithTag, ready_event: threading.Event,
                  stop_event: threading.Event, send_event_queue: queue.Queue):
        ready_event.set()
        while self.listening:
            try:
                stop_event.clear()
                ready_event.wait()
                stream = stream_loader()
                config = config_loader()
                # 使用带超时的 get 操作替代 empty 检查和 sleep
                wrapped_data = task_queue.get(timeout=0.1)
                with self.sending_lock:
                    self.sending_data_name = wrapped_data.get_name()
                    self.send_handshake(stream, config, wrapped_data, stop_event, send_event_queue)
                    if wrapped_data.get_name() != VAL_MSG_NAME:
                        file_path = wrapped_data.get_name()
                        data_bytes_size = wrapped_data.get_size()
                        self.notify_event_queue.put(Event(Evt.NOTIFY_FILE, file_path, o1=data_bytes_size))

                        send_time = cal_send_time(config, data_bytes_size)
                        send_event_queue.put(Event(Evt.SEND_FILE_START, file_path, data_bytes_size, send_time))

                        filename = os.path.basename(file_path)
                        bytes_data = wrapped_data.get_data()
                        self.send_data_bytes(stream, config, bytes_data, stop_event, send_event_queue,
                                             filename=filename)
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
                    self.notify_event_queue.put(Event(Evt.NOTIFY_FILE, None))
                    filename = os.path.basename(handshake.get(KEY_NAME))
                    bytes_size = handshake.get(KEY_SIZE)
                    send_time = config.silence_start + config.silence_stop + bytes_size * 8 / config.modem_bps
                    listen_event_queue.put(Event(Evt.RECV_FILE_START, filename, bytes_size, send_time))

                    f_path = os.path.join(RECEIVE_FOLDER, filename)
                    with open(f_path, "wb+") as temp_file:
                        recv_ok = self.receive_data_signal(stream, config, dst=temp_file, filename=filename,
                                                           stop_event=stop_event)
                    if recv_ok:
                        origin_filename = filename.split(MULTI_FILE_SEP)[0]
                        if origin_filename in self.merge_files_record:
                            tails_status = self.merge_files_record[origin_filename]
                            tails_status[filename] = True
                            tails_status_values = list(tails_status.values())
                            log.info(f"Received multipart file {filename}, "
                                     f"progress: {sum(tails_status_values)}/{len(tails_status_values)}")
                            if all(tails_status_values):
                                f_path_list = [os.path.join(RECEIVE_FOLDER, _) for _ in tails_status.keys()]
                                merge_files(f_path_list, os.path.join(RECEIVE_FOLDER, origin_filename))
                        listen_event_queue.put(Event(Evt.RECV_FILE_FINISH, f_path))
                    else:
                        listen_event_queue.put(Event(Evt.FILE_FAIL, f_path))
                        if stream is self.r_stream and not stop_event.is_set():
                            self.negot_local_recv_fail(filename)
                            if MULTI_FILE_SEP in filename:
                                log.info(f"Request resend multipart file: {filename}")
                                self.negot_resend_whole(filename)

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
