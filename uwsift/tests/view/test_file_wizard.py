import sys
from unittest.mock import patch

import pytest
from PyQt5 import QtWidgets
from satpy import DataID
from satpy.dataset import ModifierTuple, WavelengthRange
from satpy.dataset.dataid import default_id_keys_config, minimal_default_keys_config

from uwsift.ui.pov_main_ui import Ui_MainWindow
from uwsift.view.open_file_wizard import (
    PAGE_ID_FILE_SELECTION,
    PAGE_ID_PRODUCT_SELECTION,
    NumericTableWidgetItem,
    OpenFileWizard,
)


@pytest.mark.parametrize(
    "lower, higher, exp_result",
    [
        (1, 2, True),
        (10000000, 10000001, True),
        (2, 1, False),
        (3, 2, False),
        (4, 4, False),
        (0.001, 0.01, True),
        (0.1, 0.01, False),
        (0.00001, None, True),
    ],
)
def test_numeric_table_widget_item(lower, higher, exp_result):
    ntwi_lower = NumericTableWidgetItem(f"{lower} µm", lower)
    ntwi_higher = NumericTableWidgetItem(f"{higher} µm", higher)
    assert (ntwi_lower < ntwi_higher) is exp_result


def _is_ubuntu() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                if "Ubuntu" in line:
                    return True
    except OSError:
        return False
    return False


@pytest.mark.skipif(
    (sys.version_info.major == 3 and sys.version_info.minor == 11) and _is_ubuntu(),
    reason="Test crashes for Python 3.11 in Ubuntu",
)
def test_dataset_table(qtbot):
    main_window = QtWidgets.QMainWindow()

    class Document_mock:
        def current_projection_index(self):
            return 0

    class Scene_mock:
        def __init__(self, filenames, reader_kwargs):
            pass

        def available_dataset_ids(self, reader_name=None, composites=False):
            """Generate some test data: 10 single, then 10 composite and then another 10 single datasets."""
            data_ids_single_1 = {
                DataID(id_keys=minimal_default_keys_config, name=f"TestDataID_{n:02}", resolution=1000 + n * 10)
                for n in range(0, 10)
            }
            data_ids_single_2 = {
                DataID(id_keys=minimal_default_keys_config, name=f"TestDataID_{n:02}", resolution=1000 + n * 10)
                for n in range(20, 30)
            }
            data_ids_with_composites = (
                data_ids_single_1
                | {
                    DataID(
                        id_keys=default_id_keys_config,
                        name=f"TestDataID_{n:02}",
                        resolution=1000 + n * 10,
                        calibration="reflectance",
                        wavelength=WavelengthRange(min=n / 10.0, central=n / 10.0 + 0.1, max=n / 10.0 + 0.1, unit="µm"),
                        modifiers=ModifierTuple(),
                    )
                    for n in range(10, 20)
                }
                | data_ids_single_2
            )
            if composites:
                return data_ids_with_composites
            else:
                return data_ids_single_1 | data_ids_single_2

        _readers = {}

    main_window.document = Document_mock()
    main_window.ui = Ui_MainWindow()
    main_window.ui.setupUi(main_window)
    setModel_p = patch.object(QtWidgets.QComboBox, "setModel")
    update_res_shp_spin_boxes_p = patch.object(OpenFileWizard, "_update_resampling_shape_spin_boxes")
    update_res_inf_p = patch.object(OpenFileWizard, "_update_resampling_info")
    Scene_p = patch("uwsift.util.common.Scene", Scene_mock)

    qtbot.addWidget(main_window)

    with setModel_p, update_res_shp_spin_boxes_p, update_res_inf_p, Scene_p:
        ofw = OpenFileWizard(parent=main_window)
        # TODO: Currently the test crashes with "The python test process was terminated before it could exit on its own,
        #       the process errored with: Code: 4294967295, Signal: null".
        # This happens as a consequence of the calls to self.ui.selectIDTable.setItem(idx, col_idx, item) in function
        # OpenFileWizard._populate_dataset_table when there are more columns than just one. Needs to be investigated if
        # this is a problem of PyQT5 or qtbot. Note that in the past it was also observed that after opening the
        # OpenFileWizard dialog the columns were entirely empty. Maybe this is related?
        # So, currently just set the components to be "name" only.
        ofw.config["id_components"] = ["name"]
        #
        ofw.file_page_initialized = True
        ofw._check_selected_files_for_compatibility_with_reader = lambda: None
        ofw.currentId = lambda: PAGE_ID_FILE_SELECTION
        ofw.ui.fileSelectionPage.page_complete = True
        ofw.ui.projectionComboBox.addItems(["MTG FCI FDSS"])
        ofw.ui.projectionComboBox.setCurrentIndex(0)

        ofw.file_groups = {"test_scene": {"test_reader": "path/to/test/scene"}}
        ofw.validateCurrentPage()

        ofw.initializePage(PAGE_ID_PRODUCT_SELECTION)
        sit = ofw.ui.selectIDTable

        def check_rows_displayed(item_suffix_list):
            displayed_row_num = 0
            for row_num in range(sit.rowCount()):
                if sit.isRowHidden(row_num):
                    continue
                item = sit.item(row_num, 0)
                assert item.text() == f"TestDataID_{item_suffix_list[displayed_row_num]:02}"
                displayed_row_num += 1

        assert ofw._ds_type_visible == 0  # initially all datasets should be selected to be visible

        # select only the composite datasets
        ofw.ui.productTypeSelection.setCurrentIndex(1)
        qtbot.waitUntil(lambda: ofw.ui.selectIDTable.rowCount() > 0, timeout=1000)
        check_rows_displayed(list(range(10, 20)))

        # select only the single datasets
        ofw.ui.productTypeSelection.setCurrentIndex(2)
        qtbot.waitUntil(lambda: ofw.ui.selectIDTable.rowCount() > 0, timeout=1000)
        check_rows_displayed(list(range(10)) + list(range(20, 30)))

        # now select all - single and composite datasets
        ofw.ui.productTypeSelection.setCurrentIndex(0)
        qtbot.waitUntil(lambda: ofw.ui.selectIDTable.rowCount() > 0, timeout=1000)
        check_rows_displayed(list(range(30)))
