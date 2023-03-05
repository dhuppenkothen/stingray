import copy
import os
import sys
from sys import platform
import tempfile

import numpy as np
import pytest
from astropy.io import fits

from stingray.crossspectrum import AveragedCrossspectrum
from stingray.events import EventList
from stingray.io import load_events_and_gtis, ref_mjd
from stingray.largememory import retrieveData, saveData, genDataPath, HAS_ZARR
from stingray.largememory import _retrieveDataEV, _retrieveDataLC, zarr
from stingray.lightcurve import Lightcurve
from stingray.powerspectrum import AveragedPowerspectrum


curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")

IS_LINUX = True
if not (platform == "linux" or platform == "linux2"):
    IS_LINUX = False


class TestSaveSpec(object):
    @classmethod
    def setup_class(cls):
        time = np.arange(0, 1e7)
        counts = np.random.poisson(10, time.size)
        cls.lc = Lightcurve(time, counts, skip_checks=True)

        evtimes = np.sort(np.random.uniform(0, 1e7, 10**7))
        pi = np.random.randint(0, 100, evtimes.size)
        energy = pi * 0.04 + 1.6
        cls.ev = EventList(
            time=evtimes,
            pi=pi,
            energy=energy,
            gti=[[0, 1e7]],
            dt=1e-5,
            notes="Bu",
        )

    @pytest.mark.skipif("not HAS_ZARR")
    def test_save_wrong_data(self):
        with pytest.raises(ValueError) as excinfo:
            saveData("A string", "bububu")
        assert "Invalid data: A string (str)" in str(excinfo.value)

    @pytest.mark.skipif("not HAS_ZARR")
    def test_save_lc_small(self):
        test_lc = copy.deepcopy(self.lc)
        # Make sure counts_err exists
        _ = test_lc.counts_err

        # Save small part of data, < certainly chunk_size
        _ = saveData(test_lc[:300], persist=False, chunks=100000)

    @pytest.mark.skipif("not HAS_ZARR")
    def test_save_lc(self):
        test_lc = copy.deepcopy(self.lc)
        # Make sure counts_err exists
        _ = test_lc.counts_err

        dir_name = saveData(test_lc, persist=False)

        main = os.path.join(dir_name, "main_data")
        meta = os.path.join(dir_name, "meta_data")

        errors = []

        if (
            len([f for f in os.listdir(main) if not f.startswith(".")])
            or len([f for f in os.listdir(meta) if not f.startswith(".")])
        ) == 0:
            errors.append("Lightcurve is not saved or does not exist")
        else:
            times = zarr.open_array(store=main, mode="r", path="times")[...]
            counts = zarr.open_array(store=main, mode="r", path="counts")[...]
            count_err = zarr.open_array(store=main, mode="r", path="count_err")[...]
            gti = zarr.open_array(store=main, mode="r", path="gti")[...]
            gti = gti.reshape((gti.size // 2, 2))

            dt = zarr.open_array(store=meta, mode="r", path="dt")[...]
            mjdref = zarr.open_array(store=meta, mode="r", path="mjdref")[...]
            err_dist = zarr.open_array(store=meta, mode="r", path="err_dist")[...]

            if not np.array_equal(test_lc.time, times):
                errors.append("lc.time is not saved precisely")
            if not np.array_equal(test_lc.counts, counts):
                errors.append("lc.counts is not saved precisely")
            if not np.array_equal(test_lc.counts_err, count_err):
                errors.append("lc.counts_err is not saved precisely")
            if not np.array_equal(test_lc.gti, gti):
                errors.append("lc.gti is not saved precisely")
            if not (test_lc.dt == dt):
                errors.append("lc.dt is not saved precisely")
            if not (test_lc.mjdref == mjdref):
                errors.append("lc.mjdref is not saved precisely")
            if not (test_lc.err_dist == err_dist):
                errors.append("lc.err_dist is not saved precisely")

        assert not errors, "Errors encountered:\n{}".format("\n".join(errors))

    @pytest.mark.skipif("not HAS_ZARR")
    def test_save_ev_small(self):
        # Save small part of data, < certainly chunk_size
        ev = EventList(time=np.arange(1000))
        _ = saveData(self.ev, persist=False, chunks=100000)

    @pytest.mark.skipif("not (HAS_ZARR and IS_LINUX)")
    def test_save_ev_missing_psutil_linux(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)

        _ = saveData(self.ev, persist=False)

    @pytest.mark.skipif("not HAS_ZARR or IS_LINUX")
    def test_save_ev_missing_psutil_not_linux(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)

        with pytest.warns(UserWarning) as record:
            _ = saveData(self.ev, persist=False)
        assert np.any(["will not depend on available RAM" in r.message.args[0] for r in record])

    @pytest.mark.skipif("not HAS_ZARR")
    def test_save_ev(self):
        dir_name = saveData(self.ev, persist=False)

        main = os.path.join(dir_name, "main_data")
        meta = os.path.join(dir_name, "meta_data")

        errors = []

        if (
            len([f for f in os.listdir(main) if not f.startswith(".")])
            or len([f for f in os.listdir(meta) if not f.startswith(".")])
        ) == 0:
            errors.append("EventList is not saved or does not exist")

        else:
            times = zarr.open_array(store=main, mode="r", path="times")[...]
            energy = zarr.open_array(store=main, mode="r", path="energy")[...]
            pi_channel = zarr.open_array(store=main, mode="r", path="pi_channel")[...]
            gti = zarr.open_array(store=main, mode="r", path="gti")[...]
            gti = gti.reshape((gti.size // 2, 2))
            dt = zarr.open_array(store=meta, mode="r", path="dt")[...]
            ncounts = zarr.open_array(store=meta, mode="r", path="ncounts")[...]
            mjdref = zarr.open_array(store=meta, mode="r", path="mjdref")[...]
            notes = zarr.open_array(store=meta, mode="r", path="notes")[...]

            if not np.array_equal(self.ev.time, times):
                errors.append("ev.time is not saved precisely")
            if not np.array_equal(self.ev.energy, energy):
                errors.append("ev.energy is not saved precisely")
            if not np.array_equal(self.ev.pi, pi_channel):
                errors.append("ev.pi is not saved precisely")
            if not np.array_equal(self.ev.gti, gti):
                errors.append("ev.gti is not saved precisely")
            if not np.isclose(self.ev.dt, dt):
                errors.append("ev.dt is not saved precisely")
            if not self.ev.ncounts == ncounts:
                errors.append("ev.ncounts is not saved precisely")
            if not np.isclose(self.ev.mjdref, mjdref):
                errors.append("ev.mjdref is not saved precisely")
            if not self.ev.notes == notes:
                errors.append("ev.notes is not saved precisely")

        assert not errors, "Errors encountered:\n{}".format("\n".join(errors))

    @pytest.mark.skipif("not HAS_ZARR")
    def test_save_fits_data(self):
        fname = os.path.join(datadir, "monol_testA.evt")
        dir_name = saveData(fname, persist=False)

        evtdata = load_events_and_gtis(fname, additional_columns=["PI"])
        mjdref_def = ref_mjd(fname, hdu=1)
        time_def = evtdata.ev_list
        pi_channel_def = evtdata.additional_data["PI"]
        gti_def = evtdata.gti_list
        tstart_def = evtdata.t_start
        tstop_def = evtdata.t_stop

        main = os.path.join(dir_name, "main_data")
        meta = os.path.join(dir_name, "meta_data")

        errors = []

        if (
            len([f for f in os.listdir(main) if not f.startswith(".")])
            or len([f for f in os.listdir(meta) if not f.startswith(".")])
        ) == 0:
            errors.append("EventList is not saved or does not exist")
        else:
            times = zarr.open_array(store=main, mode="r", path="times")[...]
            pi_channel = zarr.open_array(store=main, mode="r", path="pi_channel")[...]
            gti = zarr.open_array(store=main, mode="r", path="gti")[...]
            gti = gti.reshape((gti.size // 2, 2))
            tstart = zarr.open_array(store=meta, mode="r", path="tstart")[...]
            tstop = zarr.open_array(store=meta, mode="r", path="tstop")[...]
            mjdref = zarr.open_array(store=meta, mode="r", path="mjdref")[...]

            order = np.argsort(times)
            times = times[order]
            pi_channel = pi_channel[order]

            if not np.allclose(time_def, times):
                errors.append("fits.events.data.time is not saved precisely")
            if not np.array_equal(pi_channel_def, pi_channel):
                errors.append("fits.events.data.pi is not saved precisely")
            if not np.allclose(gti_def, gti):
                errors.append("fits.gti.data is not saved precisely")
            if not (tstart == tstart_def):
                errors.append("fits.events.header.tstart is not saved precisely")
            if not (tstop == tstop_def):
                errors.append("fits.events.header.tstop is not saved precisely")
            if not (mjdref == mjdref_def):
                errors.append("fits.events.header.mjdref is not saved precisely")

        assert not errors, "Errors encountered:\n{}".format("\n".join(errors))


class TestRetrieveSpec(object):
    @classmethod
    def setup_class(cls):
        time = np.arange(0, 1e7)
        counts = np.random.poisson(10, time.size)
        cls.lc = Lightcurve(time, counts, skip_checks=True)

        evtimes = np.sort(np.random.uniform(0, 1e7, 10**7))
        pi = np.random.randint(0, 100, evtimes.size)
        energy = pi * 0.04 + 1.6
        cls.ev = EventList(
            time=evtimes,
            pi=pi,
            energy=energy,
            gti=[[0, 1e7]],
            dt=1e-5,
            notes="Bu",
        )
        cls.ev_noattrs = copy.deepcopy(cls.ev)
        cls.ev_noattrs.energy = None
        cls.ev_noattrs.pi = None
        cls.ev_noattrs.mjdref = 0
        cls.ev_noattrs.gti = None
        cls.ev_noattrs.dt = 0
        cls.ev_noattrs.notes = None

        cls.lc_path = saveData(cls.lc, persist=False)
        cls.ev_path = saveData(cls.ev, persist=False)
        cls.ev_path_noattrs = saveData(cls.ev_noattrs, persist=False)

    @pytest.mark.skipif("not HAS_ZARR")
    def test_retrieve_wrong_data(self):
        with pytest.raises(ValueError) as excinfo:
            retrieveData("A string", self.lc_path)

        assert "Invalid data: A string (str)" in str(excinfo.value)

    @pytest.mark.skipif("not HAS_ZARR")
    def test_retrieve_lc_data(self):
        lc = retrieveData(data_type="Lightcurve", dir_path=self.lc_path)

        assert self.lc.__eq__(lc) is True

    @pytest.mark.skipif("not HAS_ZARR")
    def test_retrieve_ev_data(self):
        with pytest.warns(UserWarning):
            ev = retrieveData(data_type="EventList", dir_path=self.ev_path)

        gti = np.asarray([[self.ev.time[0], self.ev.time[-1]]])

        assert np.allclose(ev.time, self.ev.time)
        assert np.allclose(ev.pi, self.ev.pi)
        assert np.allclose(ev.gti, gti, atol=0.001)

    # @pytest.mark.skipif('not HAS_ZARR')
    # def test_retrieve_fits_data(self):
    #     fname = os.path.join(datadir, 'monol_testA.evt')
    #     saveData(fname, os.path.join(self.path, self.dir))
    #
    #     with fits.open(fname) as fits_data:
    #         time_def = fits_data[1].data['TIME']
    #         gti_def = fits_data[2].data
    #         for col in ['PI', 'PHA']:
    #             if col in fits_data[1].data.columns.names:
    #                 pi_channel_def = fits_data[1].data[col]
    #
    #         tstart_def = fits_data[1].header['TSTART']
    #         tstop_def = fits_data[1].header['TSTOP']
    #         mjdref_def = fits_data[1].header['MJDREF']
    #
    #
    #     times, pi_channel, gti, tstart, tstop, mjdref = retrieveData(
    #         data_type='FITS', dir_name=self.dir, path=self.path)
    #
    #     if not np.allclose(time_def, times):
    #         errors.append("fits.events.data.time is not saved precisely")
    #     if not np.array_equal(pi_channel_def, pi_channel):
    #         errors.append("fits.events.data.pi is not saved precisely")
    #     if not np.allclose(gti_def, gti):
    #         errors.append("fits.gti.data is not saved precisely")
    #     if not (tstart == tstart_def):
    #         errors.append("fits.events.header.tstart is not saved precisely")
    #     if not (tstop == tstop_def):
    #         errors.append("fits.events.header.tstop is not saved precisely")
    #     if not (mjdref == mjdref_def):
    #         errors.append("fits.events.header.mjdref is not saved precisely")
    #
    #     assert not errors, "Errors encountered:\n{}".format('\n'.join(errors))

    @pytest.mark.skipif("not HAS_ZARR")
    def test_retrieve_lc_chunk_data(self):
        lc = retrieveData(
            data_type="Lightcurve",
            dir_path=self.lc_path,
            chunk_data=True,
            chunk_size=10**5,
            offset=0,
            raw=False,
        )

        trunc_lc = self.lc.truncate(stop=10**5)

        assert trunc_lc.__eq__(lc) is True

    @pytest.mark.skipif("not HAS_ZARR")
    def test_retrieve_ev_chunk_data(self):
        maxidx = 10**5
        ev = retrieveData(
            data_type="EventList",
            dir_path=self.ev_path,
            chunk_data=True,
            chunk_size=maxidx,
            offset=0,
            raw=False,
        )

        gti = np.asarray([[self.ev.time[0], self.ev.time[-1]]])

        assert np.allclose(ev.time, self.ev.time[:maxidx])
        assert np.allclose(ev.pi, self.ev.pi[:maxidx])
        assert np.allclose(ev.gti, gti, atol=0.0001)

    @pytest.mark.skipif("not HAS_ZARR")
    def test_retrieve_lc_offset_data(self):
        lc = retrieveData(
            data_type="Lightcurve",
            dir_path=self.lc_path,
            chunk_data=True,
            chunk_size=10**5,
            offset=10**2,
            raw=False,
        )

        trunc_lc = self.lc.truncate(start=10**2, stop=10**5)

        assert trunc_lc.__eq__(lc) is True

    @pytest.mark.skipif("not HAS_ZARR")
    def test_retrieve_ev_offset_data(self):
        maxidx = 10**5
        offset = 100
        ev = retrieveData(
            data_type="EventList",
            dir_path=self.ev_path,
            chunk_data=True,
            chunk_size=maxidx,
            offset=offset,
            raw=False,
        )

        gti = np.asarray([[self.ev.time[0], self.ev.time[-1]]])

        assert np.allclose(ev.time, self.ev.time[offset:maxidx])
        assert np.allclose(ev.pi, self.ev.pi[offset:maxidx])
        assert np.allclose(ev.gti, gti, atol=0.1)

    @pytest.mark.skipif("not HAS_ZARR")
    def test_retrieve_data_bad_offset(self):
        with pytest.raises(ValueError):
            _ = retrieveData(
                data_type="EventList", dir_path=self.ev_path, chunk_data=True, offset=101010101010
            )

    @pytest.mark.skipif("not HAS_ZARR")
    def test_retrieve_ev_data_bad_offset(self):
        with pytest.raises(ValueError) as excinfo:
            _ = _retrieveDataEV(data_path=genDataPath(self.ev_path), offset=101010101010)
        assert "Offset cannot be larger than size of" in str(excinfo.value)

    @pytest.mark.skipif("not HAS_ZARR")
    def test_retrieve_lc_data_bad_offset(self):
        with pytest.raises(ValueError) as excinfo:
            _ = _retrieveDataLC(data_path=genDataPath(self.lc_path), offset=101010101010)
        assert "Offset cannot be larger than size of " in str(excinfo.value)

    @pytest.mark.skipif("not HAS_ZARR")
    def test_retrieve_ev_data_no_attrs(self):
        ev = _retrieveDataEV(data_path=genDataPath(self.ev_path_noattrs))
        assert ev.mjdref == 0
        assert ev.pi is None
        assert ev.energy is None
        assert ev.gti is None
        assert ev.notes == ""

    @pytest.mark.skipif("not HAS_ZARR")
    def test_retrieve_ev_data_raw(self):
        res = _retrieveDataEV(data_path=genDataPath(self.ev_path_noattrs), raw=True)
        times, energy, ncounts, mjdref, dt, gti, pi, notes = res
        assert mjdref == 0
        assert pi is None
        assert energy is None
        assert gti is None
        assert notes == ""


class TestChunkPS(object):
    @classmethod
    def setup_class(cls):
        maxtime = 2**21
        time = np.arange(maxtime)
        counts1 = np.random.poisson(10, time.size)
        cls.lc1 = Lightcurve(time, counts1, skip_checks=True, gti=[[0, maxtime]])

        counts2 = np.random.poisson(10, time.size)
        cls.lc2 = Lightcurve(time, counts2, skip_checks=True, gti=[[0, maxtime]])

    @pytest.mark.skipif("not HAS_ZARR")
    def test_invalid_data_to_pds(self):
        with pytest.raises(ValueError) as excinfo:
            AveragedPowerspectrum(
                [self.lc1, self.lc1], segment_size=2048, large_data=True, silent=True
            )
        assert "Invalid input data type: list" in str(excinfo.value)

    @pytest.mark.skipif("not HAS_ZARR")
    def test_events_to_cpds_unimplemented(self):
        """Large memory option not implemented for events (and maybe never will)"""
        with pytest.raises(NotImplementedError) as excinfo:
            ev1 = EventList(np.random.uniform(0, 10, 10))
            AveragedCrossspectrum(
                ev1,
                ev1,
                dt=0.01,
                segment_size=5,
                large_data=True,
                silent=True,
            )

    @pytest.mark.skipif("not HAS_ZARR")
    def test_invalid_data_to_cpds(self):
        with pytest.raises(ValueError) as excinfo:
            AveragedCrossspectrum(
                [self.lc1, self.lc1],
                [self.lc2, self.lc2],
                segment_size=4096,
                large_data=True,
                silent=True,
            )
        assert "Invalid input data type: list" in str(excinfo.value)

    @pytest.mark.skipif("not HAS_ZARR")
    def test_calc_pds(self):
        ps_normal = AveragedPowerspectrum(self.lc1, segment_size=8192, silent=True, norm="leahy")
        with pytest.warns(UserWarning) as record:
            ps_large = AveragedPowerspectrum(
                self.lc1,
                segment_size=8192,
                large_data=True,
                silent=True,
                norm="leahy",
            )
        assert np.any(["The large_data option " in r.message.args[0] for r in record])

        attrs = [
            "freq",
            "power",
            "power_err",
            "unnorm_power",
            "df",
            "n",
            "nphots",
            "gti",
            "m",
        ]
        allgood = True
        assert ps_normal.freq.size == ps_large.freq.size
        for attr in attrs:
            if not np.allclose(
                getattr(ps_normal, attr),
                getattr(ps_large, attr),
                rtol=0.1,
                atol=0.1,
            ):
                allgood = False
                print(f"Attribute = {attr} ")
                print(
                    f"Raw Array: \nOriginal: {getattr(ps_normal, attr)}, "
                    f"\nLarge: {getattr(ps_large, attr)}"
                )
                maxdev = np.amax(getattr(ps_normal, attr) - getattr(ps_large, attr))
                maxdev_percent = np.abs(
                    np.max(getattr(ps_normal, attr) - getattr(ps_large, attr)) * 100
                ) / np.max(getattr(ps_normal, attr))
                print(f"Max Deviation: {maxdev}, as %: {maxdev_percent}")
                print("\n")
        assert allgood

    @pytest.mark.skipif("HAS_ZARR")
    def test_calc_cpds_zarr_not_installed(self):
        with pytest.raises(ImportError) as excinfo:
            AveragedCrossspectrum(
                self.lc1, self.lc2, segment_size=8192, large_data=True, silent=True, legacy=True
            )
        assert "The large_data option requires zarr" in str(excinfo.value)

    @pytest.mark.skipif("HAS_ZARR")
    def test_calc_pds_zarr_not_installed(self):
        with pytest.raises(ImportError) as excinfo:
            AveragedPowerspectrum(self.lc1, segment_size=8192, large_data=True, silent=True)
        assert "The large_data option requires zarr" in str(excinfo.value)

    @pytest.mark.skipif("not HAS_ZARR")
    def test_calc_cpds(self):
        cs_normal = AveragedCrossspectrum(
            self.lc1, self.lc2, segment_size=8192, silent=True, legacy=True
        )
        with pytest.warns(UserWarning) as record:
            cs_large = AveragedCrossspectrum(
                self.lc1, self.lc2, segment_size=8192, large_data=True, silent=True
            )
            assert np.any(
                ["The large_data option and the save_all" in r.message.args[0] for r in record]
            )

        attrs = [
            "freq",
            "power",
            "unnorm_power",
            "df",
            "n",
            "nphots1",
            "nphots2",
            "m",
            "gti",
        ]
        assert cs_normal.freq.size == cs_large.freq.size

        allgood = True
        for attr in attrs:
            if not np.allclose(
                getattr(cs_normal, attr),
                getattr(cs_large, attr),
                rtol=0.1,
                atol=0.1,
            ):
                print(f"Attribute = {attr} ")
                print(
                    f"Raw Array: \nOriginal: {getattr(cs_normal, attr)}, \n"
                    f"Large: {getattr(cs_large, attr)}"
                )
                maxdev = np.amax(getattr(cs_normal, attr) - getattr(cs_large, attr))
                maxdev_percent = np.abs(
                    np.max(getattr(cs_normal, attr) - getattr(cs_large, attr)) * 100
                ) / np.max(getattr(cs_normal, attr))
                print(f"Max Deviation: {maxdev}, as %: {maxdev_percent}")
                print("\n")
                allgood = False
        assert allgood
