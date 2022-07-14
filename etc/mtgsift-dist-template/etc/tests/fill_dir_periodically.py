import os
import pathlib
import re
import shutil
import warnings
from datetime import datetime, timezone, timedelta
from time import sleep
import time
import trollsift
from dateutil import parser as dt_parser
from dateutil.relativedelta import relativedelta

from uwsift import config
from uwsift.model.catalogue import Catalogue


def gen_next_dt(initial_dt: datetime, mode: str):
    gen_dt = initial_dt
    if mode == "SEVIRI":
        offset = relativedelta(minutes=5)
    elif mode == "FCI":
        offset = relativedelta(seconds=15)
    else:
        raise AttributeError(f"Unknown mode {mode} encountered in file name.")
    while True:
        yield gen_dt
        gen_dt += offset


def insert_new_file_into_dir(file, target_dir, t_stamp_full_disk, *args, **kwargs):
    tmp_path = pathlib.Path(shutil.copy(file, target_dir))
    time_idx = -1
    time_chunk = None
    # TODO(mk): assumes "-" to be delimiter, does this generalize?, do we care?
    file_name_chunks = re.split("(-)", tmp_path.name)
    for idx, chunk in enumerate(file_name_chunks):
        try:
            dt_parser.parse(chunk)
            time_idx = idx
            time_chunk = chunk
            break
        except (TypeError, dt_parser.ParserError):
            pass
    time_len = len(time_chunk)
    new_dt = str(t_stamp_full_disk).replace("-", "").replace(" ", "").replace(":", "")[:time_len]
    file_name_chunks[time_idx] = new_dt
    new_file_name = "".join(file_name_chunks)
    new_file = os.path.join(tmp_path.parent, new_file_name)
    os.rename(tmp_path, os.path.join(tmp_path.parent, new_file_name))
    print(f"Created: {file} -> {new_file}")


def fill_dir_periodically_seviri(data_dir: str, tmp_dir: str, sleep_time: float):
    """
    Fill specified directory with existing SEVIRI test files, found in in_dir, altering the
    respective start time and end_time of every file to times correctly offset from present
    time.

    :param tmp_dir: Temporary directory to write test data to.
    :param tmp_dir: Temporary directory to write test data to.

    :param data_dir: Directory from which data to be renamed is taken.

    :param sleep_time: Time between writes to out_dir.

    Note: SEVIRI and FCI work just differently enough to be a bother, EPI, PRO & granules
          for seviri all same start_time whereas FCI has a different start_time for every file.
    """
    catalogue_config = config.get("catalogue", None)
    first_query = catalogue_config[0]
    # FIXME(mk) cast to list b/c Tuple is immutable, problematic if first entry of tuple,
    #  search_path, is a str and not a list or dict and thus immutable.
    # catalogue_args = list(Catalogue.extract_query_parameters(first_query))
    first_query["search_path"] = data_dir
    first_query["constraints"]["start_time"] = {
        "type": "datetime",
        "Y": 2019,
        "m": 10,
        "d": 21,  # TODO(mk): make this dependant on the data in data dir and not hardcoded
    }
    reader_scenes_ds_ids, readers = Catalogue.query_for_satpy_importer_kwargs_and_readers(
        *Catalogue.extract_query_parameters(first_query)
    )
    file_tuples = list(reader_scenes_ds_ids["scenes"].keys())
    file_stack = []
    for file_tup in file_tuples:
        tmp = list(file_tup)
        granules_files = list(tmp[8:])
        granules_files.extend(file_tup[:8])
        for file in granules_files:
            file_stack.append(file)

    initial_dt = datetime.now(tz=timezone.utc)
    time_gen = gen_next_dt(initial_dt, "SEVIRI")
    for idx, file in enumerate(file_stack):
        # SEVIRI: advance time if one full disk image full
        if idx % 10 == 0:
            t_stamp_full_disk = next(time_gen)
        insert_new_file_into_dir(file, tmp_dir, t_stamp_full_disk)
        sleep(sleep_time)


def fill_dir_periodically_fci(data_dir: str, tmp_dir: str, sleep_time: float, retain_n_times: int) -> None:
    """
    Fill specified directory with existing FCI test files, found in in_dir, altering the
    respective start time and end_time of every file to times correctly offset from present
    time.

    :param tmp_dir: Temporary directory to write test data to.

    :param data_dir: Directory from which data to be renamed is taken.

    :param sleep_time: Time between writes to out_dir.

    Note: SEVIRI and FCI work just differently enough to be a bother, EPI, PRO & granules
          for seviri all same start_time whereas FCI has a different start_time for every file.
    """
    try:
        # format_string = config["data_reading.fci_l1c_fdhsi.filter_patterns"][0]
        format_string = (
            "{pflag}_{location_indicator},{data_designator},MTI{spacecraft_id:1d}+{data_source}-"
            "{processing_level}-{type}-{subtype}-{coverage}-{subsetting}-{component1}-BODY-{component3}-"
            "{purpose}-{format}_{oflag}_{originator}_{processing_time:%Y%m%d%H%M%S}_{facility_or_tool}_"
            "{environment}_{start_time:%Y%m%d%H%M%S}_{end_time:%Y%m%d%H%M%S}_{processing_mode}_"
            "{special_compression}_{disposition_mode}_{repeat_cycle_in_day:>04d}_"
            "{count_in_repeat_cycle:>04d}.nc"
        )

    except Exception:
        raise ValueError("Reader not found or wrong reader config.")

    time_gen = gen_next_dt(datetime.now(tz=timezone.utc), "FCI")
    files_and_start_ts = []
    files_fullpath = []
    for subdata_dir in sorted(os.listdir(data_dir)):
        for file_name in sorted(os.listdir(data_dir+'/'+subdata_dir)):
            try:
                start_t = trollsift.parse(format_string, file_name)["start_time"]
                files_and_start_ts.append((file_name, start_t))
                files_fullpath.append(data_dir+'/'+subdata_dir+'/'+file_name)
            except Exception:
                # FIXME(mk): filter pattern does not work for TRAILER file...
                warnings.warn("FCI TRAIL files not yet supported by parser, skipping...")
                pass

    files_and_start_ts = sorted(files_and_start_ts, key=lambda tup: tup[1])
    old_files = [file for file, _ in files_and_start_ts]
    file_stack = []
    for file_name in old_files:
        sat_filename_keyvals = trollsift.parse(format_string, file_name)
        try:
            start_time = sat_filename_keyvals["start_time"]
            end_time = sat_filename_keyvals["end_time"]
            start_to_end_span = end_time - start_time
        except KeyError:
            raise KeyError(
                f"Could not match replacement fields"
                f" 'start_time', 'end_time' or 'repeat_cycle_in_day'"
                f" of filter_pattern '{format_string}'"
                f" in file name '{file_name}'."
            )
        # replace datetime
        sat_filename_keyvals["start_time"] = next(time_gen)
        sat_filename_keyvals["end_time"] = sat_filename_keyvals["start_time"] + start_to_end_span
        sat_filename_keyvals["processing_time"] = sat_filename_keyvals["start_time"] + start_to_end_span + timedelta(seconds=5)
        file_name = trollsift.compose(format_string, sat_filename_keyvals)
        file_stack.append(file_name)
    # old_files = [os.path.join(data_dir, old_file) for old_file in old_files]
    for n, (old_file, new_filename) in enumerate(zip(files_fullpath, file_stack)):
        st = time.time()
        if n >= retain_n_times:
            os.remove(os.path.join(tmp_path.parent, file_stack[n-retain_n_times]))
            print(f"Removed: {file_stack[n-retain_n_times]}")
        tmp_path = pathlib.Path(shutil.copy(old_file, tmp_dir+'/tempfile.nc'))
        os.rename(tmp_path, os.path.join(tmp_path.parent, new_filename))
        print(f"Created: {old_file} -> {new_filename}")
        elapsed_time = time.time()-st
        if elapsed_time<0:
            print("Copy is running late (longer than sleep time). Moving on right away.")
            continue
        else:
            print(f"Skipping elapsed time {elapsed_time}")
            sleep(sleep_time-elapsed_time)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Time delayed directory population.")
    parser.add_argument(
        "-i",
        "--input_directory",
        action="store",
        required=True,
        type=str,
        help="Directory to take original test data from.",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        action="store",
        required=True,
        type=str,
        help="Directory to populate with mocked test data.",
    )
    parser.add_argument(
        "-t", "--time_delay", action="store", default=5.0, type=float, help="Time between two arriving files in seconds"
    )

    parser.add_argument(
        "-r", "--retain_n_files", action="store", default=-1, type=int, help="Keep only the last n files in folder. "
                                                                             "Default is -1 which means no automatic"
                                                                             "cleanup."
    )
    args = parser.parse_args()

    out_dir = args.output_directory

    if not os.path.exists(args.input_directory):
        raise ValueError("Could not find data dir.")
    in_dir = args.input_directory
    time_delay = args.time_delay
    retain_n_files = args.retain_n_files
    fill_dir_periodically_fci(in_dir, out_dir, time_delay, retain_n_files)

    # try:
    #    reader = config["catalogue"][0]["reader"]
    # except KeyError:
    #    raise KeyError("Reader not set in config files.")
    # if reader == "seviri_l1b_hrit":
    #    fill_dir_periodically_seviri(in_dir, out_dir, time_delay)
    # elif reader == "fci_l1c_fdhsi":
    #    fill_dir_periodically_fci(in_dir, out_dir, time_delay)
    # else:
    #    raise NotImplementedError(f"Reader {reader} not supported yet")
    # print(f"\nDONE copying test data to: {out_dir}\n")
