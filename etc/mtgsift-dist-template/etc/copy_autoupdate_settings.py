#! /usr/bin/env python3
"""
Script to copy the auto update settings to the user env

"""
import sys
import os
import tempfile
import datetime
import calendar

import shutils
import distutils.dir_util as dt
from distutils.dir_util import copy_tree

CONF_ROOT_DIR   ="{}/.config/SIFT"
AUTO_UPDATE_FILE="{}/settings/config/auto_update.yaml".format(CONF_ROOT_DIR)
CATALOGUE_FILE  ="{}/settings/config/catalogue.yaml".format(CONF_ROOT_DIR)

WORK_DIR = "{}/work"
CONF_DIR = "{}/conf"
DEFAULT_AUTO_UPDATE ="{}/conf/auto_update_settings/auto_update.yaml"
DEFAULT_CATALOGUE   ="{}/conf/auto_update_settings/catalogue.yaml"

def get_home_dir_path():
    """
       Get the software root dir
    """
    mtgsift_dir = os.getenv("MTGSIFT_HOME", None)
    
    # check by default in user[HOME]
    if not mtgsift_dir:
        print("Error, no ENV variable $MTGSIFT_HOME defined. Please set the $MTGSIFT_HOME to root directory of the MTGSIFT distribution.")
        sys.exit(1)
    
    #create dir if not there
    makedirs(mtgsift_dir)
    
    return mtgsift_dir

# A UTC class.    
class UTC(datetime.tzinfo):    
    """UTC Timezone"""    
    
    def utcoffset(self, a_dt): #pylint: disable=W0613
        ''' return utcoffset '''  
        return ZERO    
    
    def tzname(self, a_dt): #pylint: disable=W0613
        ''' return tzname '''    
        return "UTC"    
        
    def dst(self, a_dt): #pylint: disable=W0613 
        ''' return dst '''      
        return ZERO  

# pylint: enable-msg=W0613    
UTC_TZ = UTC()

def str2datetime(a_str, a_pattern = '%Y%m%dT%H:%M:%SZ'):
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

def datetime2str(a_datetime, a_pattern = '%Y%m%dT%H:%M:%SZ'):
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

    #utcfromtimestamp is not working properly with a decimals.
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
   """ return the user home dir
   """
   return os.path.expanduser("~")


def get_random_name():
    """ get a random filename or dirname"""
    return next(tempfile._get_candidate_names())

def makedirs(a_path):
    """ my own version of makedir """
    
    if os.path.isdir(a_path):
        # it already exists so return
        return
    elif os.path.isfile(a_path):
        raise OSError("a file with the same name as the desired dir, '{}', already exists.".format(a_path))

    os.makedirs(a_path)


def copy_and_backup():
   """ 
      Default copy and backup
   """
   home_dir = get_home()

   mtgsift_root_dir = get_home_dir_path()
   work_dir         = WORK_DIR.format(mtgsift_root_dir)
   default_auto_update = DEFAULT_AUTO_UPDATE.format(mtgsift_root_dir)
   default_catalogue   = DEFAULT_CATALOGUE.format(mtgsift_root_dir)

   #settings directory in ~/.config/SIFT
   setting_root_dir = CONF_ROOT_DIR.format(home_dir)
   auto_update_file = AUTO_UPDATE_FILE.format(home_dir)
   catalogue_file   = CATALOGUE_FILE.format(home_dir)

   print("Make a backup copy of the auto_update settings, namely auto_update.yaml and catalogue.yaml .")

   print("Looking for auto update files in {}".format(home_dir))

   if auto_update_file and os.path.exists(auto_update_file) and os.path.isfile(auto_update_file):
      #copy the content of the setting dir in WORK_DIR
      print("Backup {} into {}".format(auto_update_file, WORK_DIR))

      makedirs(WORK_DIR)
      shutil.copy(auto_udate_file, WORK_DIR)

   if catalogue_file and os.path.exists(catalogue_file) and os.path.isfile(catalogue_file):
      #copy the content of the setting dir in WORK_DIR
      print("Backup {} into {}".format(catalogue_file, WORK_DIR))

      makedirs(WORK_DIR)
      shutil.copy(catalogue_file, WORK_DIR)

   print("Copying auto_update mode files to the user env in {}".format(home_dir))
   shutil.copy(default_catalogue, catalogue_file) 
   shutil.copy(default_auto_update, auto_update_file) 

def restore():
   """
      Restore files in setting dir of the user
   """
   home_dir = get_home()

   mtgsift_root_dir = get_home_dir_path()
   work_dir         = WORK_DIR.format(mtgsift_root_dir)
   work_auto_update = AUTO_UPDATE_FILE.format(work_dir)
   work_catalogue   = CATALOGUE_FILE.format(work_dir)

   #settings directory in ~/.config/SIFT
   setting_root_dir = CONF_ROOT_DIR.format(home_dir)
   auto_update_file = AUTO_UPDATE_FILE.format(home_dir)
   catalogue_file   = CATALOGUE_FILE.format(home_dir)

   print("Restore the auto_update settings, namely auto_update.yaml and catalogue.yaml of the user.")

   print("Looking for auto update files in {}".format(home_dir))

   if work_auto_update and os.path.exists(work_auto_update) and os.path.isfile(work_auto_update):
      #copy the content of the setting dir in WORK_DIR
      print("Restore {} into {}".format(work_auto_update, setting_root_dir))

      shutil.copy(work_udate_file, setting_root_dir)

   if work_catalogue and os.path.exists(work_catalogue) and os.path.isfile(work_catalogue):
      #copy the content of the setting dir in WORK_DIR
      print("Restore {} into {}".format(work_catalogue, setting_root_dir))

      shutil.copy(catalogue_file, setting_root_dir)

   print("Copying auto_update mode files to the user env in {}".format(setting_root_dir))
   

if __name__ == '__main__':

    if sys.argv != 1:
        print("Error. need one argument\n copy_autoupdate_settings.py (restore|update)\n - restore: restore the files saved in work.\n - update: update the files with the default auto_update.\n")
        res = 1
    else:
        if sys.argv[1] == "restore":
           restore()
           res = 0
       elif sys.argv[1] == "update":
           copy_and_backup()
           res = 0
       else:
        print("Uncorrect argument {}.\n copy_autoupdate_settings.py (restore|update)\n - restore: restore the files saved in work.\n - update: update the files with the default auto_update.\n".format(sys.argv[1])) 
        res = 1

    sys.exit(res)
