import os
import copy
import pytest
import numpy as np
from stingray.base import StingrayObject, StingrayTimeseries

_HAS_XARRAY = _HAS_PANDAS = _HAS_H5PY = True

try:
    import xarray
    from xarray import Dataset
except ImportError:
    _HAS_XARRAY = False

try:
    import pandas
    from pandas import DataFrame
except ImportError:
    _HAS_PANDAS = False

try:
    import h5py
except ImportError:
    _HAS_H5PY = False


class DummyStingrayObj(StingrayObject):
    main_array_attr = "guefus"

    def __init__(self, dummy=None):
        self.guefus = dummy
        # StingrayObject.__init__(self)


class DummyStingrayTs(StingrayTimeseries):
    main_array_attr = "time"

    def __init__(self, dummy=None):
        self.guefus = dummy
        # StingrayObject.__init__(self)


# A StingrayObject with no main_array_attr
class BadStingrayObj(StingrayObject):
    def __init__(self, dummy=None):
        self.guefus = dummy
        StingrayObject.__init__(self)


def _check_equal(so, new_so):
    for attr in ["time", "guefus", "pardulas", "panesapa"]:
        if not hasattr(so, attr):
            assert not hasattr(new_so, attr)
            continue
        so_attr = at if (at := getattr(so, attr)) is not None else []
        new_so_attr = at if (at := getattr(new_so, attr)) is not None else []

        assert np.allclose(so_attr, new_so_attr)

    for attr in ["mjdref", "pirichitus", "parafritus"]:
        if not hasattr(so, attr):
            assert not hasattr(new_so, attr)
            continue
        so_attr = getattr(so, attr)
        new_so_attr = getattr(new_so, attr)
        assert so_attr == new_so_attr


class TestStingrayObject:
    @classmethod
    def setup_class(cls):
        cls.arr = [4, 5, 2]
        sting_obj = DummyStingrayObj(cls.arr)
        sting_obj.pardulas = [3.0 + 1.0j, 2.0j, 1.0 + 0.0j]
        sting_obj.pirichitus = 4
        sting_obj.parafritus = "bonus!"
        sting_obj.panesapa = [[41, 25], [98, 3]]
        cls.sting_obj = sting_obj

    def test_preliminary(self):
        assert np.allclose(self.sting_obj.guefus, self.arr)

    def test_instantiate_without_main_array_attr(self):
        with pytest.raises(RuntimeError):
            BadStingrayObj(self.arr)

    def test_side_effects(self):
        so = copy.deepcopy(self.sting_obj)
        assert np.allclose(so.guefus, [4, 5, 2])
        so.guefus = np.random.randint(0, 4, 3)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        new_so = DummyStingrayObj([4, 5, 6])
        assert not hasattr(new_so, "panesapa")

    def test_astropy_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        assert np.allclose(so.guefus, [4, 5, 2])
        so.guefus = np.random.randint(0, 4, 3)
        # Set an attribute to a DummyStingrayObj. It will *not* be saved
        so.stingattr = DummyStingrayObj([3, 4, 5])
        ts = so.to_astropy_table()
        new_so = DummyStingrayObj.from_astropy_table(ts)
        _check_equal(so, new_so)
        assert not hasattr(new_so, "stingattr")

    @pytest.mark.skipif("not _HAS_XARRAY")
    def test_xarray_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        assert np.allclose(so.guefus, [4, 5, 2])
        so.guefus = np.random.randint(0, 4, 3)
        ts = so.to_xarray()
        new_so = DummyStingrayObj.from_xarray(ts)

        _check_equal(so, new_so)

    @pytest.mark.skipif("not _HAS_PANDAS")
    def test_pandas_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        assert np.allclose(so.guefus, [4, 5, 2])
        so.guefus = np.random.randint(0, 4, 3)
        ts = so.to_pandas()
        new_so = DummyStingrayObj.from_pandas(ts)

        _check_equal(so, new_so)

    def test_astropy_roundtrip_empty(self):
        # Set an attribute to a DummyStingrayObj. It will *not* be saved
        so = DummyStingrayObj([])
        ts = so.to_astropy_table()
        new_so = DummyStingrayObj.from_astropy_table(ts)
        assert new_so.guefus is None or len(new_so.guefus) == 0

    @pytest.mark.skipif("not _HAS_XARRAY")
    def test_xarray_roundtrip_empty(self):
        so = DummyStingrayObj([])
        ts = so.to_xarray()
        new_so = DummyStingrayObj.from_xarray(ts)

        assert new_so.guefus is None or len(new_so.guefus) == 0

    @pytest.mark.skipif("not _HAS_PANDAS")
    def test_pandas_roundtrip_empty(self):
        so = DummyStingrayObj([])
        ts = so.to_pandas()
        new_so = DummyStingrayObj.from_pandas(ts)

        assert new_so.guefus is None or len(new_so.guefus) == 0

    @pytest.mark.skipif("not _HAS_H5PY")
    def test_hdf_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        so.write("dummy.hdf5")
        new_so = so.read("dummy.hdf5")
        os.unlink("dummy.hdf5")

        _check_equal(so, new_so)

    def test_file_roundtrip_fits(self):
        so = copy.deepcopy(self.sting_obj)
        so.guefus = np.random.randint(0, 4, 3)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        so.write("dummy.fits")
        new_so = DummyStingrayObj.read("dummy.fits")
        os.unlink("dummy.fits")
        # panesapa is invalid for FITS header and got lost
        assert not hasattr(new_so, "panesapa")
        new_so.panesapa = so.panesapa
        _check_equal(so, new_so)

    @pytest.mark.parametrize("fmt", ["pickle", "ascii", "ascii.ecsv"])
    def test_file_roundtrip(self, fmt):
        so = copy.deepcopy(self.sting_obj)
        so.guefus = np.random.randint(0, 4, 3)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        so.write(f"dummy.{fmt}", fmt=fmt)
        new_so = DummyStingrayObj.read(f"dummy.{fmt}", fmt=fmt)
        os.unlink(f"dummy.{fmt}")

        _check_equal(so, new_so)


class TestStingrayTimeseries:
    @classmethod
    def setup_class(cls):
        cls.arr = [4, 5, 2]
        sting_obj = DummyStingrayTs(cls.arr)
        sting_obj.time = np.asarray([0, 1, 2])
        sting_obj.mjdref = 59777.000
        sting_obj.parafritus = "bonus!"
        sting_obj.panesapa = np.asarray([[41, 25], [98, 3]])
        sting_obj.gti = np.asarray([[-0.5, 2.5]])
        cls.sting_obj = sting_obj

    def test_astropy_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        # Set an attribute to a DummyStingrayObj. It will *not* be saved
        ts = so.to_astropy_table()
        new_so = DummyStingrayTs.from_astropy_table(ts)
        _check_equal(so, new_so)

    def test_astropy_ts_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        ts = so.to_astropy_timeseries()
        new_so = DummyStingrayTs.from_astropy_timeseries(ts)
        _check_equal(so, new_so)

    def test_shift_time(self):
        new_so = self.sting_obj.shift(1)
        assert np.allclose(new_so.time - 1, self.sting_obj.time)
        assert np.allclose(new_so.gti - 1, self.sting_obj.gti)

    def test_change_mjdref(self):
        new_so = self.sting_obj.change_mjdref(59776.5)
        assert new_so.mjdref == 59776.5
        assert np.allclose(new_so.time - 43200, self.sting_obj.time)
        assert np.allclose(new_so.gti - 43200, self.sting_obj.gti)
