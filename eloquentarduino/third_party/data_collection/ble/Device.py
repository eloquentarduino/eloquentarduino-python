import re
import logging
from bleak import BleakScanner, BleakClient


class Device:
    """
    BLE device
    """
    def __init__(self, name: str, uuid: str = None):
        """

        """
        self._name = name
        self._uuid = uuid
        self.device = None
        self.services = {}
        self.available = []

    def __str__(self) -> str:
        """
        Convert to string
        """
        return f'BLE Device(name="{self.name}", uuid="{self.uuid}")'

    def __repr__(self) -> str:
        """
        Convert to string
        """
        return str(self)

    @property
    def name(self):
        """
        Get device name
        """
        return self.device.name

    @property
    def uuid(self):
        """
        Get device uuid
        """
        return self.device.address

    @property
    def client(self) -> BleakClient:
        """
        Get Bleak client
        """
        return BleakClient(self.uuid)

    async def connect(self):
        """
        Connect to device
        """
        await self.scan()

        if self._uuid is None:
            self.device = self.find_by_name()
        else:
            self.device = self.find_by_uuid()

        assert self.device is not None, f'Cannot find device with name="{self._name}" or uuid="{self._uuid}"'

        await self.get_services()

    async def get_services(self):
        """
        Get device services
        """
        self.services = {}

        async with BleakClient(self.uuid) as client:
            services = await client.get_services()

            for service in services.services.values():
                self.services[service.uuid] = [c for c in service.characteristics]

        logging.info(f'Service tree {self.services}')

    def get_characteristic_uuid(self, query: str) -> str:
        """
        Get characteristic uuid by query
        """
        for characteristics in self.services.values():
            for characteristic in characteristics:
                if self._match(characteristic.uuid, query):
                    return characteristic.uuid

        raise KeyError(f'Characteristic "{query}" not found in device {self.name}')


    async def scan(self):
        """
        Scan BLE devices
        """
        self.available = await BleakScanner.discover()
        logging.info(f'Found {len(self.available)} devices: {[str(d) for d in self.available]}')

    def find_by_name(self):
        """
        Find device by name
        """
        try:
            return [d for d in self.available if self._match(d.name, self._name)][0]
        except IndexError:
            return None

    def find_by_uuid(self):
        """
        Find device by uuid
        """
        try:
            return [d for d in self.available if self._match(d.address, self._uuid)][0]
        except IndexError:
            return None

    def _match(self, target: str, query: str):
        """
        Test if target matches query
        """
        if '*' in query:
            try:
                return re.search(query.replace('*', '.*'), target) is not None
            except TypeError:
                return False

        return target.lower() == query.lower()
