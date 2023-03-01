#! /usr/bin/env python3
"""Script to copy the settings if they are not up to date.

if the file setting.update doesn't exist in ~user/config/SIFT/settings/config copy the default settings dir
"""
# TODO: If setting.update exists and the contained date is older than the one contained in the default settings,
#  overwrite it


import calendar
import datetime
import distutils.dir_util as dt
import os
import sys
import tempfile

CONF_ROOT_DIR = "{}/.config/SIFT"
SETTING_UPDATE_FILE = "{}/settings/settings.version".format(CONF_ROOT_DIR)


def get_home_dir_path():
    """
    Get the software root dir
    """
    sift_dir = os.getenv("SIFT_HOME", None)

    # check by default in user[HOME]
    if not sift_dir:
        print(
            "Error, no ENV variable $SIFT_HOME defined. "
            "Please set the $SIFT_HOME to root directory of the SIFT distribution."
        )
        sys.exit(1)

    # create dir if not there
    makedirs(sift_dir)

    return sift_dir


# A UTC class.
class UTC(datetime.tzinfo):
    """UTC Timezone"""

    def utcoffset(self, a_dt):  # pylint: disable=W0613
        """return utcoffset"""
        return 0

    def tzname(self, a_dt):  # pylint: disable=W0613
        """return tzname"""
        return "UTC"

    def dst(self, a_dt):  # pylint: disable=W0613
        """return dst"""
        return 0

    # pylint: enable-msg=W0613


UTC_TZ = UTC()


def str2datetime(a_str, a_pattern="%Y%m%dT%H:%M:%SZ"):
    """
    :param a_datetime: the datetime.
    :param a_pattern: the datetime string pattern to parse (default ='%Y%m%dT%H:%M:%SZ').
    :return formatted string from a datetime.
    """
    if a_str:
        dt = datetime.datetime.strptime(a_str, a_pattern)
        dt.replace(tzinfo=UTC_TZ)
        return dt

    return None


def datetime2str(a_datetime, a_pattern="%Y%m%dT%H:%M:%SZ"):
    """
    :param a_datetime: the datetime.
    :param a_pattern: the datetime string pattern to use for the conversion (default ='%Y%m%dT%H:%M:%SZ').
    :return formatted string from a datetime
    """
    if a_datetime:
        return a_datetime.strftime(a_pattern)

    return None


def e2datetime(a_epoch):
    """
    convert epoch time in datetime

    :param  long a_epoch: the epoch time to convert
    :return datetime: a datetime
    """

    # utcfromtimestamp is not working properly with a decimals.
    # use floor to create the datetime
    #    decim = decimal.Decimal('%s' % (a_epoch)).quantize(decimal.Decimal('.001'), rounding=decimal.ROUND_DOWN)

    new_date = datetime.datetime.utcfromtimestamp(a_epoch)

    return new_date


def get_utcnow_epoch():
    return datetime2e(datetime.datetime.utcnow())


def datetime2e(a_date):
    """
    convert datetime in epoch
    Beware the datetime as to be in UTC otherwise you might have some surprises
        Args:
           a_date: the datertime to convert
        Returns: a epoch time
    """
    return calendar.timegm(a_date.timetuple())


def get_home():
    """return the user home dir"""
    return os.path.expanduser("~")


def get_random_name():
    """get a random filename or dirname"""
    return next(tempfile._get_candidate_names())


def makedirs(a_path):
    """my own version of makedir"""

    if os.path.isdir(a_path):
        # it already exists so return
        return
    elif os.path.isfile(a_path):
        raise OSError("a file with the same name as the desired dir, '{}', already exists.".format(a_path))

    os.makedirs(a_path)


def run():
    """
    Default runner
    """

    home_dir = get_home()

    mtgsift_root_dir = get_home_dir_path()

    # settings directory in ~/.config/SIFT
    setting_root_dir = CONF_ROOT_DIR.format(home_dir)
    setting_file = SETTING_UPDATE_FILE.format(home_dir)

    print("Looking for settings.version file {}.".format(setting_file))

    if setting_file and os.path.exists(setting_file) and os.path.isfile(setting_file):
        # replace dir if it is an older version
        print("The settings.version file exists. Nothing to do.")
    else:
        # copy the content of the default setting dir in CONF_ROOT_DIR
        print("The file doesn't exist.")

        input_dir = "{}/resources/config/SIFT/settings".format(mtgsift_root_dir)
        output_dir = "{}/{}".format(setting_root_dir, get_random_name())

        # make the dir if necessary
        makedirs(output_dir)

        print("Copy the new settings dir in {}.".format(output_dir))

        # copy all the files in the temp dir
        dt.copy_tree(input_dir, output_dir)

        # rename the old settings dir into settings.dir
        settings_dir = "{}/settings".format(setting_root_dir)

        if os.path.exists(settings_dir) and os.path.isdir(settings_dir):
            os.rename(
                settings_dir,
                "{}/settings.old.{}".format(
                    setting_root_dir, datetime2str(e2datetime(get_utcnow_epoch()), "%Y%m%dT%H%M%SZ")
                ),
            )

        # rename new dir in settings
        os.rename(output_dir, settings_dir)


if __name__ == "__main__":
    run()
