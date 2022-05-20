import os
import sys

from win32com.client import Dispatch

shell = Dispatch("WScript.Shell")
home = sys.argv[1]
workspace = sys.argv[2]
channel = sys.argv[3]
env_name = sys.argv[4]
pkg_name = sys.argv[5]

update_path = os.path.join(home, "Desktop", "Update SIFT.lnk")
print("Creating update shortcut: %s" % (update_path,))
update_shortcut = shell.CreateShortcut(update_path)
update_shortcut.Targetpath = "cmd.exe"
update_shortcut.Arguments = f"/c activate {env_name} && conda update -y -c {channel} {pkg_name} && pause && deactivate"
update_shortcut.save()

run_path = os.path.join(home, "Desktop", "Run SIFT.lnk")
print("Creating run shortcut: %s" % (run_path,))
run_shortcut = shell.CreateShortcut(run_path)
run_shortcut.Targetpath = "cmd.exe"
run_shortcut.Arguments = f"/c activate {env_name} && python -m {pkg_name} -w {workspace} && pause && deactivate"
run_shortcut.save()

# rsync_shortcut = shell.CreateShortcut(os.path.join(home, "Desktop", "Sync AHI Data.lnk"))
# rsync_shortcut.Targetpath = """C:\\cwrsync\\rsync_sift_ahi.bat"""
# rsync_shortcut.save()
