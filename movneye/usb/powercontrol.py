import os
import time
os.system('sudo -vS')

# VERY IMPORTANT NOTE:
# When running the functions in this file, the sudo password should be required only at the time this file is imported
# and will no longer be required later on. To do so, you must edit the sudoers file in order to extend the inactivity
# timeout (i.e. the time period in which the sudo password will be remembered in the session and after which Linux will
# prompt for your password again, which is usually around 5 minutes):
# Enter the visudo command "sudo visudo". An editor will open up. Look for the line "Defaults        env_reset"
# and replace it with "Defaults        env_reset, timestamp_timeout=-1" (meaning that the timeout will never expire
# during the session, i.e. after it was entered the first time).
# Finally, save changes and close (press CTRL + X, then press y and ENTER).


# --------------------------------------------- Deactivate/Reactivate USB ---------------------------------------------

def inactivate_usb_timeout(usb_id: str,
                           inactive_period: int):
    """This function controls the power of a device connected to the USB: it deactivates the USB and then
    reactivates it after some time.
    Args:
        usb_id (str, required): the ID of a USB-connected device, with {bus}-{port+level} format (e.g. you can take this
                                info using the function in movneye.usb.idfinder).
        inactive_period (int, required): time period (in seconds) in which the USB port should be deactivated.
    """
    p1 = deactivate_usb(usb_id)
    time.sleep(inactive_period)
    p2 = activate_usb(usb_id)
    if all([p1, p2]):
        return True
    return False


def deactivate_usb(usb_id: str):
    # This works for kernels >= 2.6.38
    p = os.system(f'sudo bash -c "echo 0 > /sys/bus/usb/devices/{usb_id}/power/autosuspend_delay_ms; echo auto > /sys/bus/usb/devices/{usb_id}/power/control"')
    # # This should work for previous kernels instead
    # p1 = os.system(f'sudo bash -c "echo 0 > /sys/bus/usb/devices/{usb_id}/power/autosuspend"')
    # p2 = os.system(f'sudo bash -c "echo auto > /sys/bus/usb/devices/{usb_id}/power/level"')
    if p != 0:
        return False
    return True


def activate_usb(usb_id: str):
    # This works for kernels >= 2.6.38
    p = os.system(f'sudo bash -c "echo on > /sys/bus/usb/devices/{usb_id}/power/control"')
    # # This should work for previous kernels instead
    # p = os.system(f'sudo bash -c "echo on > /sys/bus/usb/devices/{usb_id}/power/level"')
    if p != 0:
        return False
    return True


# def deactivate_usb_old(usb_id: str):
#     # This works for kernels >= 2.6.38
#     p1 = subprocess.run(['echo', '"0"', '>', f'"/sys/bus/usb/devices/{usb_id}/power/autosuspend_delay_ms"'],
#                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     p2 = subprocess.run(['echo', '"auto"', '>', f'"/sys/bus/usb/devices/{usb_id}/power/control"'],
#                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     # # This should work for previous kernels instead
#     # p1 = subprocess.run(['echo', '"0"', '>', f'"/sys/bus/usb/devices/{usb_id}/power/autosuspend"'],
#     #                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     # p2 = subprocess.run(['echo', '"auto"', '>', f'"/sys/bus/usb/devices/{usb_id}/power/level"'],
#     #                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if any([p1.returncode != 0, p2.returncode != 0]):
#         return False
#     return True
#
#
# def activate_usb_old(usb_id: str):
#     # This works for kernels >= 2.6.38
#     p = subprocess.run(['echo', '"on"', '>', f'"/sys/bus/usb/devices/{usb_id}/power/control"'],
#                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     # # This should work for previous kernels instead
#     # p = subprocess.run(['echo', '"on"', '>', f'"/sys/bus/usb/devices/{usb_id}/power/level"'],
#     #                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if p.returncode != 0:
#         return False
#     return True


# ------------------------------------------------- Unbind/Rebind USB -------------------------------------------------

def unbind_usb(usb_id: str):
    p = os.system(f'sudo bash -c "echo {usb_id} > /sys/bus/usb/drivers/usb/unbind"')
    if p != 0:
        return False
    return True


def bind_usb(usb_id: str):
    p = os.system(f'sudo bash -c "echo {usb_id} > /sys/bus/usb/drivers/usb/bind"')
    if p != 0:
        return False
    return True
