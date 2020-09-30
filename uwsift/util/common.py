from datetime import  datetime
from typing import List, Set

from satpy import DataID, Scene


def is_datetime_format(format_str):
    format_result = datetime.today().strftime(format_str)
    return format_str != format_result


def create_scenes(scenes: dict, file_groups: dict) -> List[DataID]:
    """Create Scene objects for the selected files."""
    all_available_products: Set[DataID] = set()
    for group_id, file_group in file_groups.items():
        scn = scenes.get(group_id)
        if scn is None:
            # need to create the Scene for the first time
            # file_group includes what reader to use
            # NOTE: We only allow a single reader at a time
            scenes[group_id] = scn = Scene(filenames=file_group)

            # WORKAROUND: to decompress compressed SEVIRI HRIT files, an environment variable
            # needs to be set. Check if decompression might have introduced errors when using
            # the specific reader and loading a file with compression flag set.
            # NOTE: in case this workaround-check fails data cannot be loaded in SIFT although
            # creating the scene might have succeeded!
            compressed_seviri = False
            from satpy.readers.hrit_base import get_xritdecompress_cmd
            # TODO: Scene may not provide information about reader in the
            # future - here the "protected" variable '_readers' is used as
            # workaround already
            for r in scn._readers.values():
                # only perform check when using a relevant reader, so that this is not triggered
                # mistakenly when another reader uses the same meta data key for another purpose
                if r.name in ['seviri_l1b_hrit']:
                    for fh in r.file_handlers.values():
                        for fh2 in fh:
                            if fh2.mda.get('compression_flag_for_data'):
                                compressed_seviri = True
            if compressed_seviri:
                get_xritdecompress_cmd()
            # END OF WORKAROUND

        all_available_products.update(scn.available_dataset_ids())

    # update the widgets
    return sorted(all_available_products)