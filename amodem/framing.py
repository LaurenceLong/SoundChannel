import binascii
import functools
import itertools
import logging
import struct
from math import erfc

import numpy as np
import reedsolo

from . import common

log = logging.getLogger(__name__)

# 创建RS编码器
NUM_SYMBOLS = 32
rs_codec = reedsolo.RSCodec(NUM_SYMBOLS)  # number of ecc symbols (you can repair nsym/2 errors and nsym erasures.


def ber_mqam(snr_db, M):
    # 根据snr重新协商纠错数量
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10.0)
    # Calculate BER for M-QAM
    ber = (2 * (np.sqrt(M) - 1) / (np.sqrt(M) * np.log2(M))) * \
          erfc(np.sqrt(3 * snr_linear / (2 * (M - 1))))
    return ber


def set_rs_codec(snr_db):
    global NUM_SYMBOLS
    global rs_codec


def encode_with_rs(data):
    # 确保数据是bytearray类型
    if not isinstance(data, bytearray):
        data = bytearray(data)
    # 编码数据
    encoded = rs_codec.encode(data)
    return encoded


def decode_with_rs(encoded_data):
    try:
        # 解码数据
        decoded, _, _ = rs_codec.decode(encoded_data)
        return decoded
    except reedsolo.ReedSolomonError as e:
        # import traceback
        # traceback.print_exc()
        return encoded_data


def encode_pack(data):
    encoded = encode_with_rs(data)
    return encoded


def decode_pack(encoded_data, chunk_size):
    chunk = bytearray(itertools.islice(encoded_data, chunk_size))
    if len(chunk) < chunk_size:
        raise ValueError(f'Incomplete frame, length {len(chunk)} < {chunk_size}(required)')

    decoded = decode_with_rs(chunk)
    return iter(decoded)


def _checksum_func(x):
    return binascii.crc32(bytes(x))


class Checksum:
    fmt = '>L'  # unsigned longs (32-bit)
    size = struct.calcsize(fmt)

    def encode(self, payload, cut_eof=False, is_last_block=False):
        checksum = _checksum_func(payload)
        if cut_eof and is_last_block:
            checksum = _checksum_func(struct.pack('>I', checksum))
        encoded = struct.pack(self.fmt, checksum) + payload
        return encoded

    def decode(self, data, cut_eof=False):
        received, = struct.unpack(self.fmt, bytes(data[:self.size]))
        payload = data[self.size:]
        expected = _checksum_func(payload)
        if cut_eof:
            eof_detected = received == _checksum_func(struct.pack('>I', expected))
            valid_fail = received != expected and not eof_detected
        else:
            eof_detected = False
            valid_fail = received != expected
        if valid_fail:
            log.warning('Invalid checksum: %08x != %08x', received, expected)
            raise ValueError('Invalid checksum')
        log.debug('Good checksum: %08x', received)
        return payload, eof_detected


class Framer:
    chunk_size = 255
    unencrypted_size = chunk_size - NUM_SYMBOLS
    block_size = unencrypted_size - 1 - 4  # 1 bytes length, 4 bytes crc
    prefix_fmt = '>B'
    uint_32_fmt = '>L'
    prefix_len = struct.calcsize(prefix_fmt)
    checksum = Checksum()

    EOF = b''

    def __init__(self):
        self.frame_id = 0

    def _pack(self, block, padded_size=None, cut_eof=False, is_last_block=False, use_fid=False):
        frame = self.checksum.encode(block, cut_eof=cut_eof, is_last_block=is_last_block)
        frame_id = self.frame_id
        self.frame_id += 1
        if not use_fid:
            packed = bytearray(struct.pack(self.prefix_fmt, len(frame)) + frame)
        else:
            # 尾部加frame_id
            packed = bytearray(
                struct.pack(self.prefix_fmt, len(frame)) + frame + struct.pack(self.uint_32_fmt, frame_id))

        if padded_size is not None:
            current_length = len(packed)
            if current_length > padded_size:
                raise ValueError(f"Packed data length ({current_length}) exceeds target length ({padded_size})")

            padding_length = padded_size - current_length
            packed.extend(b'\x00' * padding_length)
        packed = encode_pack(packed)
        return packed, frame_id

    def encode(self, data, cut_eof=False, use_fid=False):
        if not use_fid:
            iterator = common.iterate(data=data, size=self.block_size, func=bytearray, truncate=False)
        else:
            # 4 bytes frame_id
            iterator = common.iterate(data=data, size=self.block_size - 4, func=bytearray, truncate=False)
        prev_block = next(iterator, None)
        for current_block in iterator:
            packed, frame_id = self._pack(block=prev_block, padded_size=self.unencrypted_size, use_fid=use_fid)
            if not use_fid:
                yield packed
            else:
                yield packed, frame_id

            prev_block = current_block

        if prev_block is not None:
            packed, frame_id = self._pack(block=prev_block, padded_size=self.unencrypted_size, cut_eof=cut_eof,
                                          is_last_block=cut_eof, use_fid=use_fid)
            if not use_fid:
                yield packed
            else:
                yield packed, frame_id

        if not cut_eof:
            # 添加EOF块
            packed, frame_id = self._pack(block=self.EOF, padded_size=self.unencrypted_size)
            if not use_fid:
                yield packed
            else:
                yield packed, frame_id

    def decode(self, data, cut_eof=False, raise_err=True, use_fid=False):
        data = iter(data)
        while True:
            try:
                pack = decode_pack(data, self.chunk_size)
                length, = _take_fmt(pack, self.prefix_fmt)
                frame = _take_len(pack, length)
                frame_id, = _take_fmt(pack, self.uint_32_fmt)
                block, eof_detected = self.checksum.decode(frame, cut_eof=cut_eof)
                if block == self.EOF:
                    log.debug('EOF frame detected')
                    return

                if not use_fid:
                    yield block
                else:
                    yield block, frame_id

                if eof_detected:
                    log.debug('End frame detected')
                    return

                self.frame_id = frame_id
            except Exception as e:
                if raise_err:
                    raise e

                frame_id = self.frame_id
                self.frame_id += 1
                if not use_fid:
                    yield b''
                else:
                    yield b'', frame_id


def _take_fmt(data, fmt):
    length = struct.calcsize(fmt)
    chunk = bytearray(itertools.islice(data, length))
    if len(chunk) < length:
        raise ValueError('missing prefix data')
    return struct.unpack(fmt, bytes(chunk))


def _take_len(data, length):
    chunk = bytearray(itertools.islice(data, length))
    if len(chunk) < length:
        raise ValueError('missing payload data')
    return chunk


def chain_wrapper(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        return itertools.chain.from_iterable(result)

    return wrapped


class BitPacker:
    byte_size = 8

    def __init__(self):
        bits_list = []
        for index in range(2 ** self.byte_size):
            bits = [index & (2 ** k) for k in range(self.byte_size)]
            bits_list.append(tuple((1 if b else 0) for b in bits))

        self.to_bits = dict((i, bits) for i, bits in enumerate(bits_list))
        self.to_byte = dict((bits, i) for i, bits in enumerate(bits_list))


@chain_wrapper
def encode(data, framer=None, cut_eof=False, use_fid=False):
    converter = BitPacker()
    framer = framer or Framer()
    if not use_fid:
        for frame in framer.encode(data, cut_eof=cut_eof, use_fid=use_fid):
            for byte in frame:
                yield converter.to_bits[byte]
    else:
        for frame, frame_id in framer.encode(data, cut_eof=cut_eof, use_fid=use_fid):
            for byte in frame:
                yield converter.to_bits[byte], frame_id


@chain_wrapper
def _to_bytes(bits):
    converter = BitPacker()
    for chunk in common.iterate(data=bits, size=8,
                                func=tuple, truncate=True):
        yield [converter.to_byte[chunk]]


def decode_frames(bits, framer=None, cut_eof=False, raise_err=True, use_fid=False):
    framer = framer or Framer()
    if not use_fid:
        for frame in framer.decode(_to_bytes(bits), cut_eof=cut_eof, raise_err=raise_err, use_fid=use_fid):
            yield bytes(frame)
    else:
        for frame, frame_id in framer.decode(_to_bytes(bits), cut_eof=cut_eof, raise_err=raise_err, use_fid=use_fid):
            yield bytes(frame), frame_id
