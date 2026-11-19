"""Cross-format consistency tests for multi-frame semantics.

ImageIO exposes a uniform interface for multi-frame ImageResources
(multi-page files, animations, and videos):

- frames are selected by a zero-based ``index`` in the order in which they
  appear in the file (``read``/``properties``/``metadata``),
- the number of frames is reported consistently via
  ``properties(index=...).n_images``,
- timing metadata of animations is reported consistently via the global
  metadata keys ``duration`` (total seconds) and ``fps``,
- out-of-range indices (including negative ones) raise an ``IndexError``,
- single-frame images keep their historic behavior.

These tests pin the semantics across all multi-frame plugins so that the
behavior cannot drift apart again.
"""

import numpy as np
import pytest

import imageio.v3 as iio


def _write_multiframe_files(tmp_path, test_images):
    """Create the multi-frame files used by the parametrized tests."""

    files = {}

    # APNG written via pillow (from the GIF test image)
    apng_path = tmp_path / "multiframe.apng"
    iio.imwrite(apng_path, iio.imread(test_images / "newtonscradle.gif"), duration=50)
    files["apng"] = apng_path

    # multi-series TIFF written via tifffile
    tiff_path = tmp_path / "multiframe.tiff"
    frames = [np.full((8, 6), idx, dtype=np.uint8) for idx in range(3)]
    iio.imwrite(tiff_path, frames, plugin="tifffile", is_batch=True)
    files["tiff"] = tiff_path

    # NPZ with two arrays (read via the legacy plugin wrapper)
    npz_path = tmp_path / "multiframe.npz"
    iio.imwrite(
        npz_path,
        [np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)],
    )
    files["npz"] = npz_path

    return files


# (id, file-key, plugin, n_frames)
MULTIFRAME_CASES = [
    ("gif-pillow", "gif", "pillow", 36),
    ("gif-opencv", "gif", "opencv", 36),
    ("apng-pillow", "apng", "pillow", 36),
    ("multipage-tiff-pillow", "multipage_tiff", "pillow", 2),
    ("tiff-tifffile", "tiff", "tifffile", 3),
    ("mp4-pyav", "mp4", "pyav", 36),
    ("mp4-ffmpeg", "mp4", "FFMPEG", 36),
    ("spe", "spe", "SPE", 2),
    ("npz-legacy", "npz", None, 2),
]

# cases that report standardized timing metadata (animations/videos)
ANIMATION_CASES = [
    ("gif-pillow", "gif", "pillow", 36),
    ("apng-pillow", "apng", "pillow", 36),
    ("mp4-pyav", "mp4", "pyav", 36),
    ("mp4-ffmpeg", "mp4", "FFMPEG", 36),
]


@pytest.fixture(scope="module")
def multiframe_files(tmp_path_factory, test_images):
    tmp_path = tmp_path_factory.mktemp("multiframe")
    files = _write_multiframe_files(tmp_path, test_images)
    files["gif"] = test_images / "newtonscradle.gif"
    files["multipage_tiff"] = test_images / "multipage.tif"
    files["mp4"] = test_images / "realshort.mp4"
    files["spe"] = test_images / "test_000_.SPE"
    return files


def _open_kwargs(case, multiframe_files):
    _, file_key, plugin, _ = case
    kwargs = {"uri": multiframe_files[file_key]}
    if plugin is not None:
        kwargs["plugin"] = plugin
    return kwargs


@pytest.mark.parametrize(
    "case", MULTIFRAME_CASES, ids=[case[0] for case in MULTIFRAME_CASES]
)
def test_sequential_frame_access(multiframe_files, case):
    """read(index=i) yields the i-th frame in file order for every format."""

    *_, n_frames = case
    kwargs = _open_kwargs(case, multiframe_files)

    frames = list(iio.imiter(**kwargs))
    assert len(frames) == n_frames

    for idx in range(n_frames):
        frame = iio.imread(index=idx, **kwargs)
        np.testing.assert_array_equal(frame, frames[idx])


@pytest.mark.parametrize(
    "case", MULTIFRAME_CASES, ids=[case[0] for case in MULTIFRAME_CASES]
)
def test_frame_count_is_consistent(multiframe_files, case):
    """n_images matches the number of frames that can actually be read."""

    case_id, _, _, n_frames = case
    kwargs = _open_kwargs(case, multiframe_files)

    if case_id == "mp4-ffmpeg":
        # the legacy FFMPEG plugin cannot know the frame count without
        # decoding the whole video and reports ``inf`` ("unknown")
        pytest.skip("FFMPEG cannot determine its frame count up-front.")

    props = iio.improps(index=..., **kwargs)
    assert props.is_batch
    assert props.n_images == n_frames
    assert props.shape[0] == n_frames


@pytest.mark.parametrize(
    "case", ANIMATION_CASES, ids=[case[0] for case in ANIMATION_CASES]
)
def test_duration_metadata_is_consistent(multiframe_files, case):
    """Animations report ``duration`` (seconds) and ``fps`` uniformly."""

    *_, n_frames = case
    kwargs = _open_kwargs(case, multiframe_files)

    meta = iio.immeta(**kwargs)

    assert isinstance(meta["duration"], float)
    assert meta["duration"] > 0
    assert meta["fps"] > 0
    # duration and fps describe the same (full) animation
    assert meta["duration"] == pytest.approx(n_frames / meta["fps"], rel=0.05)


@pytest.mark.parametrize(
    "case", MULTIFRAME_CASES, ids=[case[0] for case in MULTIFRAME_CASES]
)
def test_out_of_range_index_fails_explicitly(multiframe_files, case):
    """Out-of-range indices raise IndexError (not format-specific errors)."""

    *_, n_frames = case
    kwargs = _open_kwargs(case, multiframe_files)

    with pytest.raises(IndexError):
        iio.imread(index=n_frames, **kwargs)

    with pytest.raises(IndexError):
        iio.imread(index=n_frames + 100, **kwargs)

    with pytest.raises(IndexError):
        iio.imread(index=-1, **kwargs)

    with pytest.raises(IndexError):
        iio.improps(index=n_frames, **kwargs)


@pytest.mark.parametrize(
    "case",
    [case for case in MULTIFRAME_CASES if case[2] in ("pillow", "tifffile", "pyav")],
    ids=[
        case[0]
        for case in MULTIFRAME_CASES
        if case[2] in ("pillow", "tifffile", "pyav")
    ],
)
def test_out_of_range_metadata_fails_explicitly(multiframe_files, case):
    """Metadata access with an out-of-range index raises IndexError."""

    *_, n_frames = case
    kwargs = _open_kwargs(case, multiframe_files)

    with pytest.raises(IndexError):
        iio.immeta(index=n_frames, **kwargs)

    with pytest.raises(IndexError):
        iio.immeta(index=-1, **kwargs)


@pytest.mark.parametrize("plugin", ["pillow", "opencv"])
def test_single_frame_behavior_unchanged(test_images, plugin):
    """Single-frame images behave as before (no batch semantics leak in)."""

    png = test_images / "chelsea.png"

    # default read returns a single image (no batch axis)
    image = iio.imread(png, plugin=plugin)
    assert image.shape == (300, 451, 3)

    # index=0 selects the (only) image
    np.testing.assert_array_equal(iio.imread(png, index=0, plugin=plugin), image)

    # iteration yields exactly one frame
    frames = list(iio.imiter(png, plugin=plugin))
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], image)

    # properties of a single image are not a batch
    props = iio.improps(png, plugin=plugin)
    assert props.shape == (300, 451, 3)
    assert props.n_images is None
    assert not props.is_batch

    # out-of-range access fails explicitly
    with pytest.raises(IndexError):
        iio.imread(png, index=1, plugin=plugin)

    with pytest.raises(IndexError):
        iio.imread(png, index=-1, plugin=plugin)


def test_gif_default_behavior_unchanged(test_images):
    """The plugin-chosen defaults for multi-frame files are unchanged."""

    # pillow reads all frames of a GIF by default
    gif = test_images / "newtonscradle.gif"
    assert iio.imread(gif, plugin="pillow").shape == (36, 150, 200, 3)

    # ... and so does stacking explicitly
    stacked = iio.imread(gif, index=..., plugin="pillow")
    np.testing.assert_array_equal(iio.imread(gif, plugin="pillow"), stacked)
