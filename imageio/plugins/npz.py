# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Read/Write NPZ files.

Backend: `Numpy <https://numpy.org/doc/stable/reference/generated/numpy.savez.html>`_

NPZ is a file format by numpy that provides storage of array data using gzip
compression. This imageio plugin supports data of any shape, and also supports
multiple images per file. The whole-array read/write API loads/stores all data
at once. In addition, the plugin exposes a *streaming, chunked* read API
(:meth:`Reader.iter_chunks`, :meth:`Reader.read_chunk` and
:class:`NpzStreamReader`) that reads an array in user-sized slices along its
leading axis, so that long series and large arrays no longer have to fit into
memory. The chunked path decodes the same bytes as the whole-read path, hence
``np.concatenate(list(reader.iter_chunks(chunk_size, index)))`` is pixel-for-
pixel identical to ``reader.get_data(index)``. There is no support for meta
data.

See the BSDF format for a similar (but more fully featured) format.

Parameters
----------
None

Notes
-----
This format is not available on Pypy.

"""

import numpy as np

from ..core import Format
from ._npz_streaming import NpzStreamReader


class NpzFormat(Format):
    """See :mod:`imageio.plugins.npz`"""

    def _can_read(self, request):
        # We support any kind of image data
        return request.extension in self.extensions

    def _can_write(self, request):
        # We support any kind of image data
        return request.extension in self.extensions

    # -- reader

    class Reader(Format.Reader):
        def _open(self):
            # Load npz file, which provides another file like object
            self._npz = np.load(self.request.get_file())
            assert isinstance(self._npz, np.lib.npyio.NpzFile)
            # Get list of names, ordered by name, but smarter
            self._names = sorted(self._npz.files, key=lambda x: x.split("_")[-1])
            # Lazily created streaming reader, used only by the chunked API.
            self._stream = None

        def _close(self):
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            self._npz.close()

        def _get_length(self):
            return len(self._names)

        def _get_data(self, index):
            # Get data
            if index < 0 or index >= len(self._names):
                raise IndexError("Index out of range while reading from nzp")
            im = self._npz[self._names[index]]
            # Return array and empty meta data
            return im, {}

        def _get_meta_data(self, index):
            # Get the meta data for the given index
            raise RuntimeError("The npz format does not support meta data.")

        # -- streaming / chunked API (additive, whole-read API unchanged) --

        def _get_stream(self):
            if self._stream is None:
                # Re-open the archive through the ZIP machinery. The NpzFile
                # above owns the whole-read path; this ZipFile owns the
                # chunked path and is closed in _close(). np.load() above may
                # have advanced the request's file pointer, so rewind first
                # to make the temp-file copy (if any) contain the full data.
                request_file = self.request.get_file()
                try:
                    request_file.seek(0)
                except (OSError, AttributeError):  # pragma: no cover
                    pass
                self._stream = NpzStreamReader(
                    self.request.get_local_filename(), names=self._names
                )
            return self._stream

        def iter_chunks(self, chunk_size, index=0):
            """Yield the array at ``index`` as chunks along its leading axis.

            Each yielded array contains at most ``chunk_size`` elements of the
            leading axis (the first dimension); the remaining dimensions are
            identical to the whole array. Chunks are read from disk one at a
            time, so peak data memory stays bounded by ``chunk_size`` times
            the per-element size, independently of the total array size.

            Parameters
            ----------
            chunk_size : int
                Maximum number of leading-axis elements per chunk. Must be a
                positive integer; any value is valid, including values larger
                than the leading axis (a single chunk is yielded then).
            index : int, optional
                Image/member index inside the NPZ archive. Defaults to 0.

            Yields
            ------
            chunk : numpy.ndarray
                Arrays that concatenate back into the whole array::

                    whole = np.concatenate(
                        list(reader.iter_chunks(chunk_size, index))
                    )

            See Also
            --------
            read_chunk
            NpzStreamReader
            """

            self._checkClosed()
            yield from self._get_stream().iter_chunks(chunk_size, index)

        def read_chunk(self, index, start, stop):
            """Random-access read of the ``[start:stop]`` leading-axis slice.

            Parameters
            ----------
            index : int
                Image/member index inside the NPZ archive.
            start : int
                First leading-axis index of the slice (inclusive).
            stop : int
                Last leading-axis index of the slice (exclusive).

            Returns
            -------
            chunk : numpy.ndarray
                Array equivalent to ``whole[start:stop]`` where ``whole`` is
                the array returned by :meth:`get_data` for ``index``.
            """

            self._checkClosed()
            return self._get_stream().read_chunk(index, start, stop)

        def get_stream_reader(self):
            """Return a reusable :class:`NpzStreamReader` for this archive.

            The returned object stays valid until the reader is closed and
            offers :meth:`~NpzStreamReader.iter_chunks`,
            :meth:`~NpzStreamReader.read_chunk`,
            :meth:`~NpzStreamReader.iter_all_chunks` and shape/dtype access
            without ever loading a whole member array.
            """

            self._checkClosed()
            return self._get_stream()

    # -- writer

    class Writer(Format.Writer):
        def _open(self):
            # Npz is not such a great format. We cannot stream to the file.
            # So we remember all images and write them to file at the end.
            self._images = []

        def _close(self):
            # Write everything
            np.savez_compressed(self.request.get_file(), *self._images)

        def _append_data(self, im, meta):
            self._images.append(im)  # discart meta data

        def set_meta_data(self, meta):
            raise RuntimeError("The npz format does not support meta data.")
