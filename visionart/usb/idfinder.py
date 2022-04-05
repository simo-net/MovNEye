import subprocess
from collections import Counter
from visionart.utils.add2os import all_equal


def find_usb_id(bus_number: str or int = None, device_address: str or int = None,
                vendor_id: str = None, product_id: str = None,
                manufacturer: str = None, product: str = None, serial_number: str = None):
    """Returns the USB id (in the form of a string as {bus}-{port+level}) of the device with the given input specifics."""

    specs = {'busnum': bus_number, 'devnum': device_address,
            'idVendor': vendor_id, 'idProduct': product_id,
            'manufacturer': manufacturer, 'product': product, 'serial': serial_number}
    usb_id = []
    for key, val in specs.items():
        if val:
            try:
                outs = subprocess.check_output(f'grep "{val}" /sys/bus/usb/devices/*/{key}',
                                               shell=True, stderr=subprocess.PIPE).decode()
                for o in outs.split('\n')[:-1]:
                    usb_id.append(o.split('/')[-2])
            except subprocess.CalledProcessError:
                pass

    counts_usb_ids = Counter(sorted(usb_id))
    if bus_number is not None:
        counts_usb_ids = {x: counts_usb_ids[x] for x in counts_usb_ids.keys() if x[0] == str(bus_number)}

    if not counts_usb_ids:
        raise Exception('No USB found with the given specifics.')
    if len(counts_usb_ids) > 1 and all_equal(counts_usb_ids.values()):
        raise Exception(f'There are {len(counts_usb_ids)} USBs found with the given specifics. Cannot choose a single one.')
    return sorted(counts_usb_ids.items(), key=lambda x: x[1])[-1][0]
