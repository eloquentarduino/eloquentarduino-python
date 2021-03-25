from serial import Serial
from time import time, sleep


class SerialMonitor:
    """Interact with the board via Serial"""
    def __init__(self, project):
        self.project = project

    def write(self, message, **kwargs):
        """
        Write message to serial
        """
        with self.open(**kwargs) as serial:
            serial.write(message)

    def stream(self, timeout=60, **kwargs):
        """
        Dump serial data
        :param timeout:
        """
        self.project.assert_name()
        start = time()

        with Serial(self.project.board.port, self.project.board.baud_rate, timeout=1, **kwargs) as serial:
            while time() - start < timeout:
                try:
                    char = serial.read().decode('utf-8')
                    if char:
                        print(char, end='')
                except UnicodeDecodeError:
                    pass

    def read(self, timeout=60, **kwargs):
        """
        Read from serial monitor
        """
        self.project.assert_name()
        start = time()
        buffer = ''

        with Serial(self.project.board.port, self.project.board.baud_rate, timeout=1, **kwargs) as serial:
            while time() - start < timeout:
                try:
                    char = serial.read().decode('utf-8')
                    if char:
                        buffer += char
                        self.project.logger.progress(char)
                except UnicodeDecodeError:
                    pass
        return buffer

    def read_until(self, pattern, timeout=60, **kwargs):
        """
        Read serial until a given pattern matches
        :param pattern:
        :param timeout:
        :param kwargs:
        :return:
        """
        self.project.assert_name()
        start = time()
        buffer = ''

        with Serial(self.project.board.port, self.project.board.baud_rate, timeout=1, **kwargs) as serial:
            while time() - start < timeout:
                try:
                    char = serial.read().decode('utf-8')
                    buffer += char
                    if buffer.endswith(pattern):
                        break
                except UnicodeDecodeError:
                    pass
        return buffer

    def capture_samples(self, dest, samples, append=True, dump=True, interval=0, **kwargs):
        """
        Capture the given number of samples and save them to a file in the current project
        :param dest: destination file name
        :param samples: number of samples to capture
        :param append: wether to append samples to file or overwrite existing data
        :param dump: wether to dump output to the console
        :param interval: time to wait between samples
        :param kwargs: arguments for the serial port
        :return:
        """
        self.project.assert_name()
        assert isinstance(dest, str) and len(dest) > 0, 'dest CANNOT be empty'
        assert samples > 0, 'samples MUST be grater than 0'

        with Serial(self.project.board.port, self.project.board.baud_rate, **kwargs) as serial:
            self.project.logger.debug('Serial port %s opened', self.project.board.port)
            with self.project.files.open('data', dest, mode=('a' if append else 'w')) as file:
                for i in range(samples):
                    self.project.logger.debug('%d/%d Requesting sample... ', i + 1, samples)
                    serial.write(b'capture')
                    reply = serial.readline().decode('utf-8').strip()
                    if reply:
                        file.write(reply)
                        file.write('\n')
                        self.project.logger.debug('OK')
                    else:
                        self.project.logger.warning('Empty reply')
                    # sleep between samples
                    if interval > 0:
                        sleep(interval)
        if dump:
            return self.project.files.cat('data', dest)

    def capture_streaming(self, dest, samples, delimiter=',', append=True, dump=True, timeout=60, serial_timeout=5, **kwargs):
        """Capture the given number of values and save them to a file in the current project"""
        self.project.assert_name()
        assert isinstance(dest, str) and len(dest) > 0, 'dest CANNOT be empty'
        assert samples > 0, 'samples MUST be grater than 0'

        # list of allowed characters
        alphabet = '-0123456789.\n%s' % delimiter

        with Serial(self.project.board.port, self.project.board.baud_rate, timeout=serial_timeout, **kwargs) as serial:
            with self.project.files.open('data', dest, mode=('a' if append else 'w')) as file:
                self.project.logger.info('Starting streaming acquisition... ')
                start_time = time()
                buffer = ''

                while True:
                    char = serial.read().decode('utf-8')

                    if len(char) == 1 and char in alphabet:
                        # when delimiter is found, check if we have a number
                        if char == delimiter or char == '\n':
                            if len(buffer) > 0:
                                try:
                                    float(buffer)
                                    file.write(buffer)
                                    self.project.logger.progress('.')
                                    samples -= 1

                                    if samples == 0:
                                        break
                                    else:
                                        # write delimiter or newline
                                        file.write(char)
                                except ValueError:
                                    self.project.logger.error('ValueError %s', buffer)
                                buffer = ''
                        # append character to buffer
                        else:
                            buffer += char
                    # abort on timeout
                    if time() - start_time > timeout:
                        raise RuntimeError('Timeout')

                file.write('\n')
                self.project.logger.info('DONE')

        if dump:
            self.project.files.cat('data/%s' % dest)

    def open(self, **kwargs):
        """
        Open serial port
        """
        return Serial(self.project.board.port, self.project.board.baud_rate, **kwargs)


