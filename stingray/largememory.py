import os
import random
import string
import warnings

import numpy as np
from astropy import log
from astropy.io import fits

import stingray

from .events import EventList
from .gti import cross_two_gtis
from .io import high_precision_keyword_read
from .lightcurve import Lightcurve
from .utils import genDataPath

HAS_ZARR = False
try:
    import zarr

    HAS_ZARR = True
    from numcodecs import Blosc
except ImportError:
    zarr = None
    pass

__all__ = ["createChunkedSpectra", "saveData", "retrieveData"]


def _saveChunkLC(lc, dir_name, chunks):
    """
    Save Lightcurve in chunks on disk.

    Parameters
    ----------
    lc: :class:`stingray.Lightcurve` object
        Lightcurve to be saved

    dir_name: string
        Top Level diretory name where Lightcurve is to be saved

    chunks: int
        The number of elements per chunk
    """
    # Creating a Nested Store and multiple groups for temporary saving
    store = zarr.NestedDirectoryStore(dir_name)
    lc_data_group = zarr.group(store=store, overwrite=True)
    main_data_group = lc_data_group.create_group("main_data", overwrite=True)
    meta_data_group = lc_data_group.create_group("meta_data", overwrite=True)

    compressor = Blosc(cname="lz4", clevel=1, shuffle=-1)  # Optimal

    main_data_group.create_dataset(
        name="times",
        data=lc.time,
        compressor=compressor,
        overwrite=True,
        chunks=(chunks,),
    )

    main_data_group.create_dataset(
        name="counts",
        data=lc.counts,
        compressor=compressor,
        overwrite=True,
        chunks=(chunks,),
    )

    if lc._counts_err is not None:
        main_data_group.create_dataset(
            name="count_err",
            data=lc.counts_err,
            compressor=compressor,
            overwrite=True,
            chunks=(chunks,),
        )

    main_data_group.create_dataset(name="gti", data=lc.gti.flatten(), overwrite=True)

    meta_data_group.create_dataset(name="dt", data=lc.dt, compressor=compressor, overwrite=True)

    meta_data_group.create_dataset(
        name="err_dist",
        data=lc.err_dist,
        compressor=compressor,
        overwrite=True,
    )

    meta_data_group.create_dataset(
        name="mjdref", data=lc.mjdref, compressor=compressor, overwrite=True
    )


def _saveChunkEV(ev, dir_name, chunks):
    """
    Save EventList in chunks on disk.

    Parameters
    ----------
    ev: :class:`stingray.events.EventList` object
        EventList to be saved

    dir_name: string
        Top Level diretory name where EventList is to be saved

    chunks: int
        The number of elements per chunk

    Raises
    ------
    ValueError
        If there is no data being saved
    """
    # Creating a Nested Store and multiple groups for temporary saving
    store = zarr.NestedDirectoryStore(dir_name)
    ev_data_group = zarr.group(store=store, overwrite=True)
    main_data_group = ev_data_group.create_group("main_data", overwrite=True)
    meta_data_group = ev_data_group.create_group("meta_data", overwrite=True)

    compressor = Blosc(cname="lz4", clevel=1, shuffle=-1)

    if ev.time is not None and (ev.time.all() or ev.time.size != 0):
        main_data_group.create_dataset(
            name="times",
            data=ev.time,
            compressor=compressor,
            overwrite=True,
            chunks=(chunks,),
        )

    if ev.energy is not None and (ev.energy.all() or ev.energy.size != 0):
        main_data_group.create_dataset(
            name="energy",
            data=ev.energy,
            compressor=compressor,
            overwrite=True,
            chunks=(chunks,),
        )

    if ev.pi is not None and (ev.pi.all() or ev.pi.size != 0):
        main_data_group.create_dataset(
            name="pi_channel",
            data=ev.pi,
            compressor=compressor,
            overwrite=True,
            chunks=(chunks,),
        )

    if ev.gti is not None and (ev.gti.all() or ev.gti.shape[0] != 0):
        main_data_group.create_dataset(name="gti", data=ev.gti.flatten(), overwrite=True)

    if ev.dt != 0:
        meta_data_group.create_dataset(name="dt", data=ev.dt, compressor=compressor, overwrite=True)

    if ev.ncounts:
        meta_data_group.create_dataset(
            name="ncounts",
            data=ev.ncounts,
            compressor=compressor,
            overwrite=True,
        )

    if ev.notes:
        meta_data_group.create_dataset(
            name="notes", data=ev.notes, compressor=compressor, overwrite=True
        )

    meta_data_group.create_dataset(
        name="mjdref", data=ev.mjdref, compressor=compressor, overwrite=True
    )


def _saveFITSZarr(f_name, dir_name, chunks):
    """
    Read a FITS file and save it for further processing.

    Parameters
    ----------
    f_name: string
        The name of file with which object was saved

    dir_name: string
        The name of the top level directory where the file is to be stored

    chunks: int
        The number of elements per chunk
    """

    compressor = Blosc(cname="lz4", clevel=1, shuffle=-1)

    store = zarr.NestedDirectoryStore(dir_name)
    fits_data_group = zarr.group(store=store, overwrite=True)
    main_data_group = fits_data_group.create_group("main_data", overwrite=True)
    meta_data_group = fits_data_group.create_group("meta_data", overwrite=True)

    with fits.open(f_name, memmap=True) as fits_data:
        for HDUList in fits_data:
            if HDUList.name == "EVENTS":
                times = HDUList.data["TIME"]
                chunks = times.size if times.size < chunks else chunks

                main_data_group.create_dataset(
                    name="times",
                    data=times,
                    compressor=compressor,
                    overwrite=True,
                    chunks=(chunks,),
                )

                for col in ["PI", "PHA"]:
                    if col in HDUList.data.columns.names:
                        main_data_group.create_dataset(
                            name=f"{col.lower()}_channel",
                            data=HDUList.data[col],
                            compressor=compressor,
                            overwrite=True,
                            chunks=(chunks,),
                        )

                meta_data_group.create_dataset(
                    name="tstart",
                    data=HDUList.header["TSTART"],
                    compressor=compressor,
                    overwrite=True,
                )

                meta_data_group.create_dataset(
                    name="tstop",
                    data=HDUList.header["TSTOP"],
                    compressor=compressor,
                    overwrite=True,
                )

                meta_data_group.create_dataset(
                    name="mjdref",
                    data=high_precision_keyword_read(HDUList.header, "MJDREF"),
                    compressor=compressor,
                    overwrite=True,
                )

            elif HDUList.name == "GTI":
                # TODO: Needs to be generalized
                start, stop = HDUList.data["START"], HDUList.data["STOP"]
                gti = np.array(list(zip(start, stop)))
                main_data_group.create_dataset(
                    name="gti",
                    data=gti.flatten(),
                    compressor=compressor,
                    overwrite=True,
                )


def saveData(data, persist=False, dir_name=None, chunks=None):
    """
    Saves Lightcurve/EventList or any such data in chunks to disk.

    Parameters
    ----------
    data: :class:`stingray.Lightcurve` or :class:`stingray.events.EventList` object or string
        Data to be stored on the disk.

    persist: bool
        If the data is to be stored on the disk permanently.

    dir_name: string, optional
        Name of top level directory where data is to be stored, by default randomNameGenerate()

    chunks: int, optional
        Length of data chunks in number of bins. If None, it is calculated
        based on the system resources

    Returns
    -------
    string
        Name of top level directory where data is to be stored

    Raises
    ------
    ValueError
        If data is not a Lightcurve or EventList
    """
    from sys import platform

    HAS_PSUTIL = False
    try:
        import psutil

        HAS_PSUTIL = True
    except ImportError:
        if not (platform == "linux" or platform == "linux2"):
            warnings.warn(
                "The chunk size will not depend on available RAM and will slowdown execution."
            )

    if not persist:
        import tempfile

        dir_name = tempfile.mkdtemp()
    else:
        dir_name = (
            "".join(random.choices(string.ascii_letters + string.digits, k=10))
            if dir_name is None
            else dir_name
        )

    if chunks is None:
        ideal_chunk, safe_chunk = 8388608, 4193404

        if HAS_PSUTIL:
            free_m = (psutil.virtual_memory().available + psutil.swap_memory().free) / 10**9
            chunks = ideal_chunk if free_m >= 10.0 else safe_chunk
        else:
            if platform == "linux" or platform == "linux2":
                free_m = int(os.popen("free -t -m").readlines()[-1].split()[-1])
                chunks = ideal_chunk if free_m >= 10000 else safe_chunk
            else:
                chunks = safe_chunk

    if isinstance(data, Lightcurve):
        if data.time.size < chunks:
            chunks = data.time.size
        _saveChunkLC(data, dir_name, chunks)

    elif isinstance(data, EventList):
        if not (data.time is not None and (data.time.all() or data.time.size != 0)):
            raise ValueError(("The EventList passed is empty and hence cannot be saved"))

        if data.time.size > 0 and data.time.size < chunks:
            chunks = data.time.size

        _saveChunkEV(data, dir_name, chunks)

    elif os.path.exists(data) and os.stat(data).st_size > 0:
        _saveFITSZarr(data, dir_name, chunks)

    else:
        raise ValueError((f"Invalid data: {data} ({type(data).__name__})"))

    return dir_name


def _retrieveDataLC(data_path, chunk_size=0, offset=0, raw=False):
    """
    Retrieve data from stored Lightcurve on disk.

    Parameters
    ----------
    data_path: list
        Path to datastore

    chunk_size: int, optional
        Size of data to be retrieved, default 0

    offset: int, optional
        Offset or start element to read the array from, default 0

    raw: bool, optional
        Only to be used for if raw memory mapped zarr arrays are to be obtained, default False

    Returns
    -------
    :class:`stingray.Lightcurve` object or tuple
        Lightcurve retrieved from store or data of Lightcurve

    Raises
    ------
    ValueError
        If offset provided is larger than size of array.
    """
    times = zarr.open_array(store=data_path[0], mode="r", path="times")
    counts = zarr.open_array(store=data_path[0], mode="r", path="counts")
    try:
        count_err = zarr.open_array(store=data_path[0], mode="r", path="count_err")
    except ValueError:
        count_err = None

    gti = zarr.open_array(store=data_path[0], mode="r", path="gti")[...]
    gti = gti.reshape((gti.size // 2, 2))

    dt = zarr.open_array(store=data_path[1], mode="r", path="dt")[...]
    mjdref = zarr.open_array(store=data_path[1], mode="r", path="mjdref")[...]
    err_dist = zarr.open_array(store=data_path[1], mode="r", path="err_dist")[...]

    if raw:
        return (
            times,
            counts,
            count_err,
            gti,
            float(dt),
            str(err_dist),
            float(mjdref),
        )
    else:
        if chunk_size == 0 or chunk_size > times.size:
            chunk_size = times.size
            log.info(f"Retrieving the whole array (Chunk size > size of times)")

        if offset > times.size:
            raise ValueError((f"Offset cannot be larger than size of array {times.size}"))

        times = times.get_basic_selection(slice(offset, chunk_size))[...]
        gti_new = cross_two_gtis(
            gti,
            np.asarray([[times[0] - 0.5 * float(dt), times[-1] + 0.5 * float(dt)]]),
        )

        return Lightcurve(
            time=times,
            counts=counts.get_basic_selection(slice(offset, chunk_size))[...],
            err=count_err.get_basic_selection(slice(offset, chunk_size))[...]
            if count_err is not None
            else None,
            gti=gti_new,
            dt=float(dt),
            err_dist=str(err_dist),
            mjdref=float(mjdref),
            skip_checks=True,
        )


def _retrieveDataEV(data_path, chunk_size=0, offset=0, raw=False):
    """
    Retrieve data from stored EventList on disk.

    Parameters
    ----------
    data_path: list
        Path to datastore.

    chunk_size: int
        Size of data to be retrieved

    offset: int
        Offset or start element to read the array from

    raw: bool
        Only to be used for if raw memory mapped zarr arrays are to be obtained

    Returns
    -------
    :class:`stingray.events.EventList` object or tuple
        EventList or data of EventList retrieved from store.

    Raises
    ------
    ValueError
        If array does not exist at path

    ValueError
        If offset provided is larger than size of array.

    ValueError
        If the file to read is empty
    """
    try:
        times = zarr.open_array(store=data_path[0], mode="r", path="times")
    except ValueError:
        raise ValueError("This file contains no events.")

    try:
        energy = zarr.open_array(store=data_path[0], mode="r", path="energy")
    except ValueError:
        energy = None

    try:
        pi_channel = zarr.open_array(store=data_path[0], mode="r", path="pi_channel")
    except ValueError:
        pi_channel = None

    try:
        gti = zarr.open_array(store=data_path[0], mode="r", path="gti")[...]
        gti = gti.reshape((gti.size // 2, 2))
    except ValueError:
        gti = None

    try:
        dt = zarr.open_array(store=data_path[1], mode="r", path="dt")[...]
    except ValueError:
        dt = 0

    try:
        ncounts = zarr.open_array(store=data_path[1], mode="r", path="ncounts")[...]
    except ValueError:
        ncounts = None

    try:
        mjdref = zarr.open_array(store=data_path[1], mode="r", path="mjdref")[...]
    except ValueError:
        mjdref = 0

    try:
        notes = zarr.open_array(store=data_path[1], mode="r", path="notes")[...]
    except ValueError:
        notes = ""

    if raw:
        return (
            times,
            energy,
            int(ncounts),
            float(mjdref),
            dt,
            gti,
            pi_channel,
            str(notes),
        )
    else:
        if chunk_size == 0 or chunk_size > times.size:
            chunk_size = times.size
            log.info(f"Retrieving the whole array (Chunk size > size of times)")

        if offset > times.size:
            raise ValueError("No element read. Offset cannot be larger than size of array")

        time = times.get_basic_selection(slice(offset, chunk_size))
        energy = (
            energy.get_basic_selection(slice(offset, chunk_size)) if energy is not None else None
        )
        pi = (
            pi_channel.get_basic_selection(slice(offset, chunk_size))
            if pi_channel is not None
            else None
        )

        gti_new = None
        if gti is not None:
            gti_new = cross_two_gtis(
                gti,
                np.asarray([[times[0] - 0.5 * float(dt), times[-1] + 0.5 * float(dt)]]),
            )
            warnings.warn(
                "GTI management is tricky when partially loading event lists "
                "from ZARR arrays. Please use with care.",
                UserWarning,
            )

        if pi_channel is not None:
            pi_channel = pi_channel.get_basic_selection(slice(offset, chunk_size))
        else:
            pi_channel = None

        return EventList(
            time=time,
            energy=energy,
            ncounts=ncounts,
            mjdref=float(mjdref),
            dt=float(dt),
            gti=gti_new,
            pi=pi_channel,
            notes=str(notes),
        )


def retrieveData(data_type, dir_path, chunk_data=False, chunk_size=0, offset=0, raw=False):
    """
    Retrieves Lightcurve/EventList or any such data from disk.

    Parameters
    ----------
    data_type: string
        Type of data to retrieve i.e. Lightcurve, Eventlist, FITS data to retrieve

    dir_path: string
        Path to zarr datastore + Top level directory name for data

    chunk_data: bool, optional
        If only a chunk of data is to be retrieved, by default False

    chunk_size: int, optional
        Number of values to be retrieved, by default 0

    offset: int, optional
        Start offset from where values are to be retrieved, by default 0

    raw: bool, optional
        Only to be used for if raw memory mapped zarr arrays are to be obtained, by default False

    Returns
    -------
    :class:`stingray.events.EventList` object or :class:`stingray.Lighrcurve` object or tuple
        EventList or Lightcurve created from store or raw data

    Raises
    ------
    ValueError
        If datatype is not Lightcurve or EventList of FITS
    """
    data_path = genDataPath(dir_path)

    if chunk_data is True and offset >= chunk_size:
        raise ValueError(("offset should be less than chunk_size"))

    if data_type.lower() == "lightcurve":
        if chunk_data is True and chunk_size > 0:
            return _retrieveDataLC(data_path, int(chunk_size), int(offset), raw=raw)
        else:
            return _retrieveDataLC(data_path, raw=raw)

    elif data_type.lower() == "eventlist" or data_type.lower() == "fits":
        if chunk_data is True and chunk_size > 0:
            return _retrieveDataEV(data_path, int(chunk_size), int(offset), raw=False)
        else:
            return _retrieveDataEV(data_path, raw=raw)

    else:
        raise ValueError((f"Invalid data: {data_type} ({type(data_type).__name__})"))


def _combineSpectra(final_spectra):
    """
    Create a final spectra that is the mean of all spectra.

    Parameters
    ----------
    final_spectra: :class:`stingray.AveragedCrossspectrum/AveragedPowerspectrum' object
        Summed spectra of all spectra

    Returns
    -------
    :class:`stingray.events.EventList` object or :class:`stingray.Lighrcurve` object
        Final resulting spectra.
    """
    final_spectra.power /= final_spectra.m
    final_spectra.unnorm_power /= final_spectra.m
    final_spectra.power_err = np.sqrt(final_spectra.power_err) / final_spectra.m

    if isinstance(final_spectra, stingray.AveragedCrossspectrum) and not isinstance(
        final_spectra, stingray.AveragedPowerspectrum
    ):
        final_spectra.pds1.power /= final_spectra.m
        final_spectra.pds2.power /= final_spectra.m
        final_spectra.nphots1 /= final_spectra.m
        final_spectra.nphots2 /= final_spectra.m
    else:
        final_spectra.nphots /= final_spectra.m

    return final_spectra


def _addSpectra(final_spectra, curr_spec):
    """
    Add various Spectra(AveragedCrossspectrum/AveragedPowerspectrum) for combination.

    Parameters
    ----------
    final_spectra: object
        Final Combined AveragedCrossspectrum or AveragedPowerspectrum

    curr_spec: object
        AveragedCrossspectrum/AveragedPowerspectrum to be combined

    first_iter: bool
        Check for first iteration variable

    Returns
    -------
    final_spectrum: :class:`stingray.crossspectrum.AveragedCrossspectrum` object or :class:`stingray.powerspectrum.AveragedPowerspectrum` object
        Combined AveragedCrossspectrum/AveragedPowerspectrum
    """

    final_spectra.power += curr_spec.power * curr_spec.m
    final_spectra.unnorm_power += curr_spec.unnorm_power * curr_spec.m
    final_spectra.power_err += curr_spec.power_err**2 * curr_spec.m

    final_spectra.m += curr_spec.m

    if isinstance(final_spectra, stingray.AveragedPowerspectrum):
        final_spectra.nphots += curr_spec.nphots * curr_spec.m

    elif isinstance(final_spectra, stingray.AveragedCrossspectrum):
        final_spectra.pds1.power += curr_spec.pds1.power * curr_spec.m
        final_spectra.pds1.power += curr_spec.pds1.power * curr_spec.m

        final_spectra.nphots1 += curr_spec.nphots1 * curr_spec.m
        final_spectra.nphots2 += curr_spec.nphots2 * curr_spec.m

    return final_spectra


def _chunkLCSpec(data_path, spec_type, segment_size, norm, gti, power_type, silent):
    """
    Create a chunked spectra from Lightcurve stored on disk.

    Parameters
    ----------
    data_path: string
        Path to stored Lightcurve or EventList chunks on disk

    spec_type: string
        Type of spectra to create AveragedCrossspectrum or AveragedPowerspectrum.

    segment_size: float
        The size of each segment to average in the AveragedCrossspectrum/AveragedPowerspectrum.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}
        The normalization of the (real part of the) cross spectrum.

    gti: 2-d float array
        `[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    power_type: string
        Parameter to choose among complete, real part and magnitude of
         the cross spectrum. None for AveragedPowerspectrum

    silent: bool
        Do not show a progress bar when generating an averaged cross spectrum.
        Useful for the batch execution of many spectra

    dt1: float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where data1 or data2 are of :class:EventList

    Returns
    -------
    :class:`stingray.events.EventList` object or :class:`stingray.Lighrcurve` object
        Summed computed spectra

    Raises
    ------
    ValueError
        If spectra is not AveragedCrossspectrum or AveragedPowerspectrum

    ValueError
        If previous and current spectra frequencies are not identical
    """
    times, counts, counts_err, gti_new, dt, err_dist, mjdref = _retrieveDataLC(
        data_path[0:2], raw=True
    )
    final_gti = gti_new.reshape((gti_new.size // 2, 2))

    if spec_type.lower() == "averagedpowerspectrum":
        fin_spec = stingray.AveragedPowerspectrum()

    elif spec_type.lower() == "averagedcrossspectrum":
        fin_spec = stingray.AveragedCrossspectrum()
        (
            times_other,
            counts_other,
            counts_err_other,
            gti_other,
            dt_other,
            err_dist_other,
            mjdref_other,
        ) = _retrieveDataLC(data_path[2:4], raw=True)
        gti_new_other = gti_other.reshape((gti_other.size // 2, 2))
        final_gti = cross_two_gtis(final_gti, gti_new_other)
    else:
        raise ValueError((f"Invalid spectra-type {spec_type}"))

    nchunks_per_segment = max(int(np.rint(segment_size / dt / times.chunks[0])), 1)

    new_segment_size = nchunks_per_segment * times.chunks[0] * dt
    if not np.isclose(new_segment_size, segment_size):
        warnings.warn("Adapting segment_size to a fixed number of chunks")
        segment_size = new_segment_size

    step = times.chunks[0] * nchunks_per_segment
    prev_freq = None
    for i in range(0, times.size, step):
        nmax = i + step
        if nmax > times.size and (nmax - times.size) * dt < segment_size:
            continue
        time = times.get_basic_selection(slice(i, i + step))
        count = counts.get_basic_selection(slice(i, i + step))
        count_err = None
        if counts_err is not None:
            count_err = counts_err.get_basic_selection(slice(i, i + step))

        gti_new = np.asarray([[time[0] - 0.5 * dt, time[-1] + 0.5 * dt]])

        lc1 = Lightcurve(
            time=time,
            counts=count,
            err=count_err,
            gti=gti_new,
            err_dist=err_dist,
            mjdref=mjdref,
            dt=dt,
            skip_checks=True,
        )

        if isinstance(fin_spec, stingray.AveragedPowerspectrum):
            avg_spec = stingray.AveragedPowerspectrum(
                data=lc1,
                segment_size=segment_size,
                norm=norm,
                gti=gti_new,
                silent=True,
                large_data=False,
            )

        else:
            time_other = times_other.get_basic_selection(slice(i, i + step))
            count_other = counts_other.get_basic_selection(slice(i, i + step))
            count_err_other = None
            if counts_err_other is not None:
                count_err_other = counts_err_other.get_basic_selection(slice(i, i + step))

            gti_new_other = np.asarray([[time_other[0] - 0.5 * dt, time_other[-1] + 0.5 * dt]])

            lc2 = Lightcurve(
                time=time_other,
                counts=count_other,
                err=count_err_other,
                gti=gti_new_other,
                err_dist=err_dist_other,
                mjdref=mjdref_other,
                dt=dt_other,
                skip_checks=True,
            )

            assert np.allclose(
                gti_new, gti_new_other
            ), "The two light curves should cover identical time frames"

            avg_spec = stingray.AveragedCrossspectrum(
                data1=lc1,
                data2=lc2,
                segment_size=segment_size,
                norm=norm,
                gti=gti_new,
                power_type=power_type,
                silent=True,
                large_data=False,
            )

        # First iteration
        if i == 0:
            fin_spec = avg_spec
            prev_freq = avg_spec.freq
            continue

        # All others
        if not np.allclose(prev_freq, avg_spec.freq):
            bad = ~np.isclose(prev_freq, avg_spec.freq)
            raise ValueError(
                (f"Spectra have unequal frequencies: " f"{avg_spec.freq[bad]} vs {prev_freq[bad]}")
            )
        fin_spec = _addSpectra(fin_spec, avg_spec)
        prev_freq = avg_spec.freq

    fin_spec.gti = final_gti
    return fin_spec


# TODO: Add support for eventlist soon!
# def _chunkEVSpec(data_path, spec_type, segment_size, norm, gti, power_type,
#                  silent, dt1):
#     """
#     Create a chunked spectra from EventList stored on disk.
#
#     Parameters
#     ----------
#     data_path: string
#         Path to stored Lightcurve or EventList chunks on disk
#
#     spec_type: string
#         Type of spectra to create AveragedCrossspectrum or AveragedPowerspectrum.
#
#     segment_size: float
#         The size of each segment to average in the AveragedCrossspectrum/AveragedPowerspectrum.
#
#     norm: {``frac``, ``abs``, ``leahy``, ``none``}
#         The normalization of the (real part of the) cross spectrum
#
#     gti: 2-d float array
#         `[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
#         This choice overrides the GTIs in the single light curves. Use with
#         care!
#
#     power_type: string
#         Parameter to choose among complete, real part and magnitude of
#          the cross spectrum. None for AveragedPowerspectrum
#
#     silent: bool
#         Do not show a progress bar when generating an averaged cross spectrum.
#         Useful for the batch execution of many spectra
#
#     dt1: float
#         The time resolution of the light curve. Only needed when constructing
#         light curves in the case where data1 or data2 are of :class:EventList
#
#     Returns
#     -------
#     :class:`stingray.events.EventList` object or :class:`stingray.Lighrcurve` object
#         Summed computed spectra
#
#     Raises
#     ------
#     ValueError
#         If spectra is not AveragedCrossspectrum or AveragedPowerspectrum
#
#     ValueError
#         If previous and current spectra frequencies are not identical
#     """
#     times, energy, ncounts, mjdref, dt, gti_new, pi_channel, notes = _retrieveDataEV(
#         data_path[0:2], raw=True)
#     gti_new = gti_new.reshape((gti_new.size // 2, 2))
#
#     if spec_type.lower() == 'averagedpowerspectrum':
#         fin_spec = stingray.AveragedPowerspectrum()
#
#     elif spec_type.lower() == 'averagedcrossspectrum':
#         fin_spec = stingray.AveragedCrossspectrum()
#         times_other, energy_other, ncounts_other, mjdref_other, dt_other, gti_other, pi_channel_other = _retrieveDataEV(
#             data_path[2:4], raw=True)
#         gti_new_other = gti_other.reshape((gti_other.size // 2, 2))
#
#     else:
#         raise ValueError((f"Invalid spectra {spec_type}"))
#
#     first_iter = True
#     for i in range(times.chunks[0], times.size, times.chunks[0]):
#         gti_new = cross_two_gtis(
#             gti_new,
#             np.asarray([[
#                 times.get_basic_selection(i - times.chunks[0]) - 0.5 * dt,
#                 times.get_basic_selection(i) + 0.5 * dt
#             ]]))
#         ev1 = EventList(
#             time=times.get_basic_selection(slice(i - times.chunks[0], i))
#             if times is not None else None,
#             energy=energy.get_basic_selection(slice(i - times.chunks[0], i))
#             if energy is not None else None,
#             ncounts=ncounts,
#             mjdref=mjdref,
#             dt=dt,
#             gti=gti_new,
#             pi=pi_channel.get_basic_selection(slice(i - times.chunks[0], i))
#             if isinstance(pi_channel, zarr.core.Array) else pi_channel,
#             notes=notes)
#
#         if isinstance(fin_spec, stingray.AveragedPowerspectrum):
#             avg_spec = stingray.AveragedPowerspectrum(
#                 data=ev1,
#                 segment_size=segment_size,
#                 norm=norm,
#                 gti=gti,
#                 silent=silent,
#                 dt=dt1,
#                 large_data=False)
#
#         elif isinstance(fin_spec, stingray.AveragedCrossspectrum):
#             gti_new_other = cross_two_gtis(
#                 gti_other,
#                 np.asarray([[
#                     times_other.get_basic_selection(i - times_other.chunks[0])
#                     - 0.5 * dt_other,
#                     times_other.get_basic_selection(i) + 0.5 * dt_other
#                 ]]))
#
#             ev2 = EventList(
#                 time=times_other.get_basic_selection(slice(i - times.chunks[0], i)) if time_other is not None else None,
#                 energy=energy_other.get_basic_selection(slice(i - times.chunks[0], i)) if energy_other is not None else None,
#                 ncounts=ncounts_other,
#                 mjdref=mjdref_other,
#                 dt=dt_other,
#                 gti=gti_new_other,
#                 pi=pi_channel_other.get_basic_selection(slice(i - times.chunks[0], i)) if isinstance(pi_channel_other, zarr.core.Array) else pi_channel_other,
#                 notes=notes_other)
#
#             avg_spec = stingray.AveragedCrossspectrum(
#                 data1=ev1,
#                 data2=ev2,
#                 segment_size=segment_size,
#                 norm=norm,
#                 gti=gti,
#                 power_type=power_type,
#                 silent=silent,
#                 dt=dt1,
#                 large_data=False)
#
#         if first_iter:
#             prev_freq = avg_spec.freq
#             fin_spec = _addSpectra(fin_spec, avg_spec, first_iter)
#         else:
#             if np.array_equal(prev_freq, avg_spec.freq):
#                 fin_spec = _addSpectra(fin_spec, avg_spec, first_iter)
#                 prev_freq = avg_spec.freq
#             else:
#                 raise ValueError((
#                     f"Spectra have unequal frequencies {avg_spec.freq.shape}{prev_freq.shape}"
#                 ))
#
#         first_iter = False
#     fin_spec.gti = cross_two_gtis(gti_new, gti_new_other)
#
#     return fin_spec


def createChunkedSpectra(
    data_type,
    spec_type,
    data_path,
    segment_size,
    norm,
    gti,
    power_type,
    silent,
    dt=None,
):
    """
    Create a chunked spectra from zarr files stored on disk.

    Parameters
    ----------
    data_type: string
        Data in Lightcurve or EventList

    spec_type: string
        Type of spectra to create AveragedCrossspectrum or AveragedPowerspectrum

    data_path: list
        Path to datastore

    segment_size: float
        The size of each segment to average in the AveragedCrossspectrum/AveragedPowerspectrum

    norm: {``frac``, ``abs``, ``leahy``, ``none``}
        The normalization of the (real part of the) cross spectrum

    gti: 2-d float array
        `[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    power_type: string
        Parameter to choose among complete, real part and magnitude of
         the cross spectrum. None for AveragedPowerspectrum

    silent: bool
        Do not show a progress bar when generating an averaged cross spectrum.
        Useful for the batch execution of many spectra

    dt: float, optional
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where data1 or data2 are of :class:EventList, by default None

    Returns
    -------
    :class:`stingray.events.EventList` object or :class:`stingray.Lighrcurve` object
        Final computed spectra
    """
    if data_type.lower() == "lightcurve":
        fin_spec = _chunkLCSpec(
            data_path=data_path,
            spec_type=spec_type,
            segment_size=segment_size,
            norm=norm,
            gti=gti,
            power_type=power_type,
            silent=silent,
        )

    else:
        raise NotImplementedError("Only input light curves are allowed")

    return _combineSpectra(fin_spec)
