from time import time
from tqdm.auto import tqdm
from eloquentarduino.third_party.data_collection.ble.Device import Device
from eloquentarduino.third_party.data_collection.ble.Characteristic import Characteristic


class Collector:
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
        self.characteristics = []
        self.device = Device(name=name, uuid=address)

    def add_characteristic(self, characteristic: Characteristic):
        """
        Add BLE characteristic with features' descriptions
        """
        self.characteristics.append(characteristic)

    async def connect(self):
        """
        Connect to device
        """
        await self.device.connect()
        [characteristic.connect(self.device) for characteristic in self.characteristics]

    async def collect(self, duration):
        """
        Collect data
        """
        async with self.device.client as client:
            for characteristic in self.characteristics:
                await characteristic.start_collect(client)

            with tqdm(total=duration) as progress:
                start_time = time()
                last_delta = 0

                while time() - start_time < duration:
                    delta = time() - start_time
                    progress.update(delta - last_delta)
                    last_delta = delta

            captures = [await characteristic.stop_collect()
                        for characteristic in self.characteristics]

            return captures if len(captures) > 1 else captures[0]

