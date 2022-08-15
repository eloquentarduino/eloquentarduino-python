import re
import struct
import logging
from time import time
from tqdm.auto import tqdm
from bleak import BleakScanner, BleakClient
from eloquentarduino.third_party.data_collection.DataCapture import DataCapture


class BLECollector:
    """
    Collect data over BLE
    """

    def __init__(self, name=None, address=None, characteristic_uuid=None):
        """
        :param name: str
        :param address: str
        :param characteristic_uuid: str
        """
        assert name is not None or address is not None, 'you MUST either provide an address or a name'

        self.name = name
        self.address = address
        self.characteristic_uuid = characteristic_uuid
        self.device = self.address

    async def connect(self):
        """
        Connect to device
        """

        def _match(target, pattern):
            if pattern is None:
                return True
            if '*' in pattern:
                return re.search(pattern.replace('*', '.*'), target) is not None
            return target == pattern

        if self.address is None or '*' in self.address:
            # find device by name or address wildcard
            logging.info('Scanning BLE devices...')
            devices = await BleakScanner.discover()
            logging.info(f'Found {len(devices)} devices: {[str(d) for d in devices]}')
            devices = [d for d in devices if _match(d.address, self.address) and _match(d.name, self.name)]

            assert len(devices) > 0, 'no devices matches filters'
            assert len(devices) == 1, 'more than 1 device match filters'

            self.device = devices[0].address

        if self.characteristic_uuid is None or '*' in self.characteristic_uuid:
            # get characteristic from device
            found_match = False

            async with BleakClient(self.device) as client:
                logging.info('Querying device BLE services...')
                services = await client.get_services()
                logging.info(f'Found {len(services.services)} services')

                for service in services:
                    logging.info(f'Device service: {str(service)}')

                    for characteristic in services.characteristics.values():
                        logging.info(f' > Characteristic: {str(characteristic)}')

                        if re.search(self.characteristic_uuid.replace('*', '.*'), characteristic.uuid) is not None:
                            logging.info('Found match')
                            self.characteristic_uuid = characteristic.uuid
                            found_match = True
                            break

                    if found_match:
                        break

    async def debug_services(self):
        """
        Debug device services
        """
        async with BleakClient(self.device) as client:
            logging.info('Querying device BLE services...')
            services = await client.get_services()
            logging.info(f'Found {len(services.services)} services')

            for service in services:
                logging.info(f'Device service: {str(service)}')

                for characteristic in services.characteristics.values():
                    print(f' > Characteristic: {str(characteristic)}')

    async def collect(self, duration, feature_names, with_counter=True):
        """
        Collect data for given duration
        """
        async with BleakClient(self.device) as client:
            def parser(packet):
                packet_format = 'f' * len(feature_names) + ('f' if with_counter else '')
                return struct.unpack(packet_format, packet)

            def on_notify(_, data):
                collected.append(parser(data))

            await client.start_notify(self.characteristic_uuid, on_notify)
            start_time = time()
            last_delta = 0
            collected = []

            with tqdm(total=duration) as progress:
                while time() - start_time < duration:
                    if not client.is_connected:
                        await client.stop_notify(self.characteristic_uuid)
                        await client.connect()

                    delta = time() - start_time
                    progress.update(delta - last_delta)
                    last_delta = delta

                    if not client.is_connected:
                        continue

            try:
                await client.stop_notify(self.characteristic_uuid)
            except ValueError:
                pass

            capture = DataCapture(collected)

            if with_counter:
                rx_ratio = (capture.values[-1, 0] - capture.values[0, 0]) / len(capture.values)
                logging.info('Rx ratio: %.3f' % rx_ratio)
                capture.drop([0])

            return capture.set_names(feature_names)

    async def collect_accelerometer(self, duration, **kwargs):
        """
        Collect accelerometer data
        """
        return await self.collect(duration=duration, feature_names=['ax', 'ay', 'ax'], **kwargs)

    async def collect_accelerometer_and_gyroscope(self, duration, **kwargs):
        """
        Collect accelerometer and gyroscope data
        """
        return await self.collect(duration=duration, feature_names=['ax', 'ay', 'ax', 'gx', 'gy', 'gz'], **kwargs)
