from struct import unpack
from bleak import BleakClient
from eloquentarduino.third_party.data_collection.DataCapture import DataCapture
from eloquentarduino.third_party.data_collection.ble.Device import Device


class Characteristic:
    """

    """
    def __init__(self, name, uuid, dtype='float'):
        """

        """
        self.name = name
        self.uuid = uuid
        self.dtype = dtype or 'float'
        self.client = None
        self.features = []
        self.data = []

    @property
    def packet_format(self):
        """
        Get struct packet format for binary decoding
        """
        return ''.join([feature['struct'] for feature in self.features])

    def add_feature(self, name, dtype=None, scale=1):
        """
        Add feature to be extracted from packet
        !Note: order is important!
        """
        self.features.append({
            'name': name,
            'dtype': dtype or self.dtype,
            'struct': self.packet_format_from_dtype(dtype or self.dtype),
            'scale': scale
        })

    def connect(self, device: Device):
        """
        Look for characteristic in device
        """
        self.uuid = device.get_characteristic_uuid(self.uuid)

    async def start_collect(self, client: BleakClient):
        """
        Start collecting data
        """
        self.data = []
        self.client = client

        def on_notify(_, data):
            self.data.append(self.parse_packet(data))

        await client.start_notify(self.uuid, on_notify)

    async def stop_collect(self):
        """
        Stop collecting data
        """
        await self.client.stop_notify(self.uuid)

        return DataCapture(self.data, features=self.features)

    def packet_format_from_dtype(self, dtype):
        """
        Convert default_dtype to struct identifier.
        On a 32 bit processor, long == int and double == float
        """
        return {
            'int8': 'b',
            'uint8': 'B',
            'int16': 'h',
            'uint16': 'H',
            'int': 'i',
            'unsigned int': 'I',
            'long': 'i',
            'unsigned long': 'I',
            'float': 'f',
            'double': 'f'
        }[dtype.replace('_t', '')]

    def parse_packet(self, packet):
        """
        Parse binary packet
        """
        return unpack(self.packet_format, packet)
