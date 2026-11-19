"""Test npz plugin functionality."""

import tracemalloc

import pytest

import numpy as np

import imageio.v2 as iio
import imageio.v3 as iio3
from imageio.core import Request, IS_PYPY
from imageio.plugins._npz_streaming import NpzStreamReader
from conftest import deprecated_test


@deprecated_test
def test_npz_format(test_images):
    # Test selection
    for name in ["npz", ".npz"]:
        format = iio.formats["npz"]
        assert format.name == "NPZ"
        assert format.__module__.endswith(".npz")

    # Test cannot read
    png = test_images / "chelsea.png"
    assert not format.can_read(Request(png, "ri"))
    assert not format.can_write(Request(png, "wi"))


def test_npz_reading_writing(tmp_path):
    """Test reading and saveing npz"""

    if IS_PYPY:
        return  # no support for npz format :(

    im2 = np.ones((10, 10), np.uint8) * 2
    im3 = np.ones((10, 10, 10), np.uint8) * 3
    im4 = np.ones((10, 10, 10, 10), np.uint8) * 4

    filename1 = tmp_path / "test_npz.npz"

    # One image
    iio.imsave(filename1, im2)
    im = iio.imread(filename1)
    ims = iio.mimread(filename1)
    assert (im == im2).all()
    assert len(ims) == 1

    # Multiple images
    iio.mimsave(filename1, [im2, im2, im2])
    im = iio.imread(filename1)
    ims = iio.mimread(filename1)
    assert (im == im2).all()
    assert len(ims) == 3

    # Volumes
    iio.mvolsave(filename1, [im3, im3])
    im = iio.volread(filename1)
    ims = iio.mvolread(filename1)
    assert (im == im3).all()
    assert len(ims) == 2

    # Mixed
    W = iio.save(filename1)
    assert W.format.name == "NPZ"
    W.append_data(im2)
    W.append_data(im3)
    W.append_data(im4)
    pytest.raises(RuntimeError, W.set_meta_data, {})  # no meta data support
    W.close()
    #
    R = iio.read(filename1)
    assert R.format.name == "NPZ"
    ims = list(R)  # == [im for im in R]
    assert (ims[0] == im2).all()
    assert (ims[1] == im3).all()
    assert (ims[2] == im4).all()
    # Fail
    pytest.raises(IndexError, R.get_data, -1)
    pytest.raises(IndexError, R.get_data, 3)
    pytest.raises(RuntimeError, R.get_meta_data, None)  # no meta data support
    pytest.raises(RuntimeError, R.get_meta_data, 0)  # no meta data support


def test_npz_chunked_matches_whole_read(tmp_path):
    """Chunks reconstruct the whole array pixel-for-pixel, any chunk size."""

    if IS_PYPY:
        return  # no support for npz format :(

    rng = np.random.default_rng(0)
    arrays = [
        rng.normal(size=(10, 4)).astype(np.float32),
        rng.integers(0, 1000, size=(11, 5, 3), dtype=np.int64),
        rng.normal(size=(6, 4, 3, 2)).astype(np.float64),
        np.asfortranarray(rng.normal(size=(13, 7, 5)).astype(np.float32)),
        np.asfortranarray(rng.integers(0, 255, size=(9, 11), dtype=np.uint16)),
        rng.normal(size=(8, 3)).astype(">f4"),  # non-native byte order
        np.array([[1, 2], [3, 4]], dtype=np.bool_),
    ]
    filename = tmp_path / "chunked.npz"
    iio.mimsave(filename, arrays)
    whole = iio.mimread(filename)

    reader = iio.read(filename)
    try:
        for index, arr in enumerate(arrays):
            for chunk_size in (1, 2, 3, 5, 7, 100):
                chunks = list(reader.iter_chunks(chunk_size, index=index))

                # every chunk respects the bound and keeps trailing dims
                assert all(c.shape[0] <= chunk_size for c in chunks)
                assert all(c.shape[1:] == arr.shape[1:] for c in chunks)
                assert sum(c.shape[0] for c in chunks) == arr.shape[0]

                # chunk dtype equals the whole-read dtype (byte order too)
                assert all(c.dtype.str == whole[index].dtype.str for c in chunks)

                # pixel-for-pixel identical to the whole read
                rebuilt = np.concatenate(chunks)
                assert rebuilt.shape == arr.shape
                assert np.array_equal(rebuilt, arr)
    finally:
        reader.close()


def test_npz_chunked_random_access(tmp_path):
    """read_chunk returns exactly whole[start:stop], incl. validation."""

    if IS_PYPY:
        return

    rng = np.random.default_rng(1)
    arr = rng.integers(0, 255, size=(17, 6, 4), dtype=np.uint8)
    filename = tmp_path / "random.npz"
    iio.mimsave(filename, [arr])

    reader = iio.read(filename)
    try:
        for start, stop in [(0, 1), (2, 5), (0, 17), (16, 17)]:
            chunk = reader.read_chunk(0, start, stop)
            assert np.array_equal(chunk, arr[start:stop])
            assert chunk.shape == (stop - start, 6, 4)

        # out-of-range and inverted ranges are rejected
        with pytest.raises(IndexError):
            reader.read_chunk(0, 0, 18)
        with pytest.raises(IndexError):
            reader.read_chunk(0, 5, 4)

        # non-positive chunk sizes are rejected on first iteration
        with pytest.raises(ValueError):
            list(reader.iter_chunks(0))
        with pytest.raises(ValueError):
            list(reader.iter_chunks(-3))
    finally:
        reader.close()


def test_npz_chunked_v3_api(tmp_path):
    """The chunked API is reachable through the v3 imopen wrapper."""

    if IS_PYPY:
        return

    arr = np.arange(60, dtype=np.int32).reshape(12, 5)
    filename = tmp_path / "v3.npz"
    iio.mimsave(filename, [arr])

    with iio3.imopen(filename, "r", plugin="NPZ") as file:
        chunks = list(file.iter_chunks(4, index=0))
        assert np.array_equal(np.concatenate(chunks), arr)
        assert np.array_equal(file.read_chunk(0, 2, 4), arr[2:4])


def test_npz_chunked_file_object(tmp_path):
    """Chunked reading works for file-like resources, not only filenames."""

    if IS_PYPY:
        return

    arr = np.arange(30, dtype=np.float64).reshape(10, 3)
    filename = tmp_path / "fileobj.npz"
    iio.mimsave(filename, [arr])

    with open(filename, "rb") as fh:
        reader = iio.read(fh, ".npz")
        try:
            chunks = list(reader.iter_chunks(3, index=0))
            assert np.array_equal(np.concatenate(chunks), arr)
        finally:
            reader.close()


def test_npz_stream_reader_series_order(tmp_path):
    """iter_all_chunks covers every member in whole-read order."""

    if IS_PYPY:
        return

    arrays = [(np.arange(12, dtype=np.int16).reshape(6, 2) + i) for i in range(12)]
    filename = tmp_path / "series.npz"
    iio.mimsave(filename, arrays)
    # the authoritative member order is the one of the whole-read path
    whole = iio.mimread(filename)

    with NpzStreamReader(filename) as stream:
        assert len(stream) == len(whole)
        collected = {i: [] for i in range(len(whole))}
        for index, chunk in stream.iter_all_chunks(4):
            collected[index].append(chunk)

        for index, arr in enumerate(whole):
            assert stream.shape(index) == arr.shape
            assert stream.dtype(index) == arr.dtype
            assert np.array_equal(np.concatenate(collected[index]), arr)


def test_npz_chunked_memory_bounded(tmp_path):
    """Peak memory follows the chunk cap instead of array size."""

    if IS_PYPY:
        return

    rng = np.random.default_rng(2)
    # 96 MB array; each chunk holds 8 leading indices (~3 MB).
    arr = rng.normal(size=(256, 64, 64)).astype(np.float64)
    filename = tmp_path / "big.npz"
    np.savez_compressed(filename, x=arr)

    chunk_size = 8
    chunk_bytes = chunk_size * 64 * 64 * arr.dtype.itemsize

    tracemalloc.start()
    try:
        with NpzStreamReader(filename) as stream:
            peak = 0
            seen = 0
            for chunk in stream.iter_chunks(chunk_size):
                peak = max(peak, tracemalloc.get_traced_memory()[1])
                seen += chunk.shape[0]
                assert chunk.shape[1:] == (64, 64)
    finally:
        tracemalloc.stop()

    assert seen == arr.shape[0]
    # The whole array is never materialized: peak stays within a small
    # multiple of one chunk and far below the full array.
    assert peak < 8 * chunk_bytes
    assert peak < arr.nbytes // 4


def test_npz_chunked_one_d_and_scalar(tmp_path):
    """Leading-axis chunking also covers 1-D and 0-D (scalar) arrays."""

    if IS_PYPY:
        return

    filename = tmp_path / "edge.npz"
    np.savez(filename, scalar=np.array(3.14, dtype=np.float32), vec=np.arange(12))

    with NpzStreamReader(filename) as stream:
        assert stream.shape(0) == ()
        scalar_chunks = list(stream.iter_chunks(1, index=0))
        assert len(scalar_chunks) == 1
        assert np.array_equal(scalar_chunks[0], np.array([3.14], np.float32))

        vec = np.concatenate(list(stream.iter_chunks(5, index=1)))
        assert np.array_equal(vec, np.arange(12))


def test_npz_reader_get_stream_reader(tmp_path):
    """The v2 reader exposes a reusable NpzStreamReader."""

    if IS_PYPY:
        return

    arr = np.arange(24, dtype=np.int32).reshape(6, 4)
    filename = tmp_path / "stream_reader.npz"
    iio.mimsave(filename, [arr])

    reader = iio.read(filename)
    try:
        stream = reader.get_stream_reader()
        # same backing stream is reused
        assert reader.get_stream_reader() is stream
        assert stream.shape(0) == arr.shape
        assert stream.dtype(0) == arr.dtype
        assert np.array_equal(np.concatenate(list(stream.iter_chunks(2))), arr)
        assert np.array_equal(stream.read_chunk(0, 1, 3), arr[1:3])
    finally:
        reader.close()
