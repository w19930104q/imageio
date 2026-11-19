# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Streaming/chunked reading for NPZ files.

An NPZ file is a ZIP archive whose members are NPY files. A NPY file stores
its array as a flat, contiguous block of raw bytes (C-order or Fortran-order).
This module exploits that layout to read slices ("chunks") of an array without
ever materializing the whole array in memory.

The chunked read path decodes the exact same pixel bytes as a whole read
(:func:`numpy.load` / the ``np.savez`` round-trip); it merely performs the
C/Fortran ``reshape`` on a sub-range of the data instead of the full range.
Therefore, concatenating all chunks of an array reconstructs the whole array
pixel-by-pixel.

Peak resident data memory is bounded by one chunk plus a small constant
amount of buffering, independently of the array's total size. Only
:func:`numpy.frombuffer` and dtype/shape machinery from NumPy is used, so the
code works on NumPy 1.x and 2.x alike.
"""

import ast
import zipfile

import numpy as np


_MAGIC_PREFIX = b"\x93NUMPY"
_MAGIC_LEN = 6


class NpyHeader:
    """Parsed header of a NPY stream.

    Attributes
    ----------
    shape : tuple of int
        Shape of the array stored in the NPY stream.
    dtype : numpy.dtype
        Dtype of the stored values. The byte order is taken from the header,
        so non-native endian data is handled correctly.
    fortran_order : bool
        Whether the data block is laid out in Fortran (column-major) order.
    data_offset : int
        Byte offset, relative to the start of the NPY stream, at which the
        raw array data begins.
    data_size : int
        Size of the raw array data in bytes.
    """

    __slots__ = ("shape", "dtype", "fortran_order", "data_offset", "data_size")

    def __init__(self, shape, dtype, fortran_order, data_offset):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.fortran_order = bool(fortran_order)
        self.data_offset = int(data_offset)
        self.data_size = int(
            (np.prod(self.shape) if self.shape else 1) * self.dtype.itemsize
        )


def _read_exact(fp, size, what):
    """Read exactly ``size`` bytes from ``fp`` or raise ``EOFError``."""

    data = fp.read(size)
    if len(data) != size:
        raise EOFError(
            f"Unexpected end of NPY stream while reading {what}: "
            f"expected {size} bytes, got {len(data)}."
        )
    return data


def parse_npy_header(fp):
    """Parse a NPY header positioned at the start of a stream.

    Parameters
    ----------
    fp : file-like
        A binary stream positioned at the first byte (``\\x93NUMPY``) of a
        NPY file. The stream is left positioned at the first byte of the
        array data.

    Returns
    -------
    header : NpyHeader
        The parsed header.
    """

    magic = _read_exact(fp, _MAGIC_LEN, "magic string")
    if magic != _MAGIC_PREFIX:
        raise ValueError(
            "Not a NPY stream: bad magic string %r (expected %r)."
            % (magic, _MAGIC_PREFIX)
        )

    major, minor = fp.read(2)
    if major == 1:
        # Version 1.0: header length is an unsigned 16-bit little-endian int.
        header_len = int.from_bytes(_read_exact(fp, 2, "header length"), "little")
    elif major == 2:
        # Version 2.0: header length is an unsigned 32-bit little-endian int.
        header_len = int.from_bytes(_read_exact(fp, 4, "header length"), "little")
    else:
        raise ValueError(f"Unsupported NPY format version: {major}.{minor}")

    header_bytes = _read_exact(fp, header_len, "array header")
    # NPY headers are ASCII; strip the trailing padding/newline/spaces.
    header_dict = ast.literal_eval(header_bytes.decode("latin1").strip())

    if not isinstance(header_dict, dict):
        raise ValueError("Invalid NPY header: expected a Python dict literal.")
    for key in ("descr", "fortran_order", "shape"):
        if key not in header_dict:
            raise ValueError(f"Invalid NPY header: missing key {key!r}.")

    return NpyHeader(
        shape=header_dict["shape"],
        dtype=np.dtype(header_dict["descr"]),
        fortran_order=header_dict["fortran_order"],
        data_offset=_MAGIC_LEN + 2 + (2 if major == 1 else 4) + header_len,
    )


class _NpyEntryReader:
    """Chunked reader for a single NPY member of an NPZ archive.

    C-order data is streamed linearly in a single pass: the leading axis is
    the slowest-varying one, so each chunk occupies a contiguous run of
    bytes.

    Fortran-order data has the leading axis varying fastest, so a chunk's
    elements are interleaved with the remaining chunks. Such chunks are read
    with short seeks; when the ZIP member stream cannot seek (deflated members
    on Python < 3.13) the member is re-opened per seek, which stays linear in
    the number of bytes actually delivered but never buffers the whole array.

    In every case at most one chunk-sized array (plus stream-internal
    compression buffers, independent of array size) is held in memory.
    """

    def __init__(self, zip_file, member_name):
        self._zip = zip_file
        self._name = member_name

        fp = zip_file.open(member_name, "r")
        try:
            self._header = parse_npy_header(fp)
        finally:
            fp.close()

        # Scalar arrays (shape ()) behave like a one-element leading axis.
        self._axis_size = self._header.shape[0] if self._header.shape else 1
        # Number of trailing (non-chunked) elements per chunk-axis element.
        self._tail_shape = self._header.shape[1:]
        self._n_tail = (
            int(np.prod(self._tail_shape)) if len(self._header.shape) > 1 else 1
        )
        self._item_size = self._header.dtype.itemsize

    @property
    def header(self):
        return self._header

    @property
    def axis_size(self):
        """Length of the leading (chunk) axis."""

        return self._axis_size

    def _open_at(self, data_byte_offset):
        """Open the member and position it at a raw data byte offset.

        Falls back to re-opening the member when the stream does not support
        seeking. The NPY header is skipped on every (re)open.
        """

        fp = self._zip.open(self._name, "r")
        header = parse_npy_header(fp)
        if data_byte_offset:
            try:
                fp.seek(header.data_offset + data_byte_offset)
            except (AttributeError, OSError):
                # ZipExtFile.seek only exists on Python >= 3.13; on older
                # versions re-open the member and linearly skip the offset.
                fp.close()
                fp = self._zip.open(self._name, "r")
                parse_npy_header(fp)
                fp.read(data_byte_offset)
        return fp

    def _empty_chunk(self):
        return np.empty((0, *self._tail_shape), dtype=self._header.dtype)

    def read_chunk(self, start, stop):
        """Read leading-axis slice ``[start:stop]`` as a standalone array.

        The returned array is C-ordered with the same dtype and shape
        semantics as the corresponding slice of the whole array.
        """

        if start < 0 or stop > self._axis_size or start > stop:
            raise IndexError(
                f"Chunk range [{start}:{stop}] is outside the leading axis of "
                f"length {self._axis_size}."
            )
        if start == stop:
            return self._empty_chunk()

        width = stop - start

        if not self._header.fortran_order:
            # C-order: [start:stop] occupies one contiguous byte range.
            linear_start = start * self._n_tail
            count = width * self._n_tail
            fp = self._open_at(linear_start * self._item_size)
            try:
                raw = _read_exact(fp, count * self._item_size, "array data")
            finally:
                fp.close()
            flat = np.frombuffer(raw, dtype=self._header.dtype, count=count)
            return flat.reshape((width, *self._tail_shape), order="C")

        # Fortran-order: gather ``width`` values per trailing sub-array.
        # In F-order the leading axis varies fastest, so the linear values of
        # one chunk repeat per trailing position in blocks of ``width``.
        parts = []
        stride = self._axis_size
        for block in range(self._n_tail):
            offset = (block * stride + start) * self._item_size
            fp = self._open_at(offset)
            try:
                raw = _read_exact(fp, width * self._item_size, "array data")
            finally:
                fp.close()
            parts.append(np.frombuffer(raw, dtype=self._header.dtype, count=width))
        flat = np.concatenate(parts)

        if not self._tail_shape:
            return flat.copy()
        # Undo the F-order flattening: flat[t, i] lives at linear index
        # ``i + axis_size * t`` for i in [start, stop).
        chunk = flat.reshape((self._n_tail, width), order="C").T.reshape(
            (width, *self._tail_shape), order="F"
        )
        return np.ascontiguousarray(chunk)

    def iter_chunks(self, chunk_size):
        """Yield successive chunks of at most ``chunk_size`` leading indices.

        C-order members are consumed in a single linear pass with no seeking.
        Fortran-order members use the same seek/reopen machinery as
        :meth:`read_chunk` but walk the chunks in order.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")

        start = 0
        if not self._header.fortran_order:
            # Single linear pass; the member stream is opened once.
            fp = self._open_at(0)
            try:
                while start < self._axis_size:
                    stop = min(start + chunk_size, self._axis_size)
                    count = (stop - start) * self._n_tail
                    raw = _read_exact(fp, count * self._item_size, "array data")
                    flat = np.frombuffer(raw, dtype=self._header.dtype, count=count)
                    yield flat.reshape((stop - start, *self._tail_shape), order="C")
                    start = stop
            finally:
                fp.close()
            return

        while start < self._axis_size:
            stop = min(start + chunk_size, self._axis_size)
            yield self.read_chunk(start, stop)
            start = stop


class NpzStreamReader:
    """Streaming, chunked reader for the images of an NPZ archive.

    Whole reads still go through :class:`numpy.lib.npyio.NpzFile` (the
    existing imageio API is unchanged); this class only adds a memory-bounded
    path for users that cannot load a long series or a huge array at once.

    Parameters
    ----------
    file : path-like or file-like
        The NPZ archive, as accepted by :class:`zipfile.ZipFile`.
    names : list of str, optional
        Member names in read order. Defaults to the ordering used by
        imageio's whole-read NPZ path (sorted by the substring after the
        last underscore).
    """

    def __init__(self, file, names=None):
        self._owns_zip = not isinstance(file, zipfile.ZipFile)
        self._zip = (
            file if isinstance(file, zipfile.ZipFile) else zipfile.ZipFile(file, "r")
        )
        zip_names = set(self._zip.namelist())
        if names is None:
            # Match the ordering of imageio's legacy NPZ reader, which sorts
            # members on the substring following the last underscore. This
            # keeps the chunked member order identical to the whole-read
            # order even for 10+ members.
            resolved = sorted(
                self._zip.namelist(), key=lambda name: name.split("_")[-1]
            )
        else:
            # ``NpzFile.files`` strips the trailing ``.npy`` while the ZIP
            # members keep it; accept either spelling and resolve to the
            # actual member name.
            resolved = []
            for name in names:
                if name in zip_names:
                    resolved.append(name)
                elif name + ".npy" in zip_names:
                    resolved.append(name + ".npy")
                else:
                    raise KeyError(f"There is no item named {name!r} in the archive")
        self._names = resolved
        self._entries = {}

    # -- context manager / lifecycle -------------------------------------

    def close(self):
        if self._owns_zip and self._zip is not None:
            self._zip.close()
        self._zip = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown noise
            pass

    def __len__(self):
        return len(self._names)

    # -- entry handling ---------------------------------------------------

    def names(self):
        """Return the ordered list of NPZ member names."""

        return list(self._names)

    def _entry(self, index):
        if index < 0 or index >= len(self._names):
            raise IndexError(
                f"Index {index} out of range for NPZ with {len(self._names)} "
                "member(s)."
            )
        name = self._names[index]
        if name not in self._entries:
            self._entries[name] = _NpyEntryReader(self._zip, name)
        return self._entries[name]

    def shape(self, index=0):
        """Shape of the array stored at ``index``."""

        return self._entry(index).header.shape

    def dtype(self, index=0):
        """Dtype of the array stored at ``index``."""

        return self._entry(index).header.dtype

    # -- chunked API ------------------------------------------------------

    def chunk_count(self, chunk_size, index=0):
        """Number of chunks that ``iter_chunks`` yields for ``index``."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        axis = self._entry(index).axis_size
        return int(np.ceil(axis / chunk_size)) if axis else 0

    def read_chunk(self, index, start, stop):
        """Read the ``[start:stop]`` slice of the leading axis at ``index``.

        Random access; safe to call in any order.
        """

        return self._entry(index).read_chunk(start, stop)

    def iter_chunks(self, chunk_size, index=0):
        """Yield chunks of the array at ``index`` along its leading axis.

        Each yielded array contains at most ``chunk_size`` elements of the
        leading axis. Concatenating the chunks
        (``np.concatenate(list(reader.iter_chunks(...)))``) reconstructs the
        whole array.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        return self._entry(index).iter_chunks(chunk_size)

    def iter_all_chunks(self, chunk_size):
        """Yield ``(index, chunk)`` for every member in archive order.

        Only one member stream is open at any time, so a long series is read
        incrementally instead of being loaded wholesale.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        for index in range(len(self._names)):
            for chunk in self._entry(index).iter_chunks(chunk_size):
                yield index, chunk
