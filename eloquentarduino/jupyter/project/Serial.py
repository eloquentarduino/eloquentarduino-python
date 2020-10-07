from serial import Serial
from time import time, sleep


class SerialMonitor:
    """Interact with the board via Serial"""
    def __init__(self, project):
        self.project = project

    def read(self, timeout=60, dump=True, **kwargs):
        """Read from serial monitor"""
        self.project.assert_name()
        start = time()
        buffer = ''
        with Serial(self.project.board.port, self.project.board.baud_rate, timeout=1, **kwargs) as serial:
            while time() - start < timeout:
                char = serial.read().decode('utf-8')
                buffer += char
                if dump and char:
                    self.project.log(char, end='')
        return buffer

    def capture_samples(self, dest, samples, append=True, dump=True, interval=0, **kwargs):
        """Capture the given number of samples and save them to a file in the current project"""
        self.project.assert_name()
        assert isinstance(dest, str) and len(dest) > 0, 'dest CANNOT be empty'
        assert samples > 0, 'samples MUST be grater than 0'
        with Serial(self.project.board.port, self.project.board.baud_rate, **kwargs) as serial:
            with self.project.files.open('data', dest, mode=('a' if append else 'w')) as file:
                for i in range(samples):
                    self.project.log('[%d/%d] Requesting sample... ' % (i + 1, samples), end='')
                    serial.write(b'capture')
                    reply = serial.readline().decode('utf-8').strip()
                    if reply:
                        file.write(reply)
                        file.write('\n')
                        self.project.log('OK')
                    else:
                        self.project.log('Empty reply')
                    # sleep between samples
                    if interval > 0:
                        sleep(interval)
        if dump:
            self.project.files.cat('data/%s' % dest)

    def capture_streaming(self, dest, samples, delimiter=',', append=True, dump=True, timeout=60, serial_timeout=5, **kwargs):
        """Capture the given number of values and save them to a file in the current project"""
        self.project.assert_name()
        assert isinstance(dest, str) and len(dest) > 0, 'dest CANNOT be empty'
        assert samples > 0, 'samples MUST be grater than 0'
        # list of allowed characters
        alphabet = '0123456789.\n%s' % delimiter
        with Serial(self.project.board.port, self.project.board.baud_rate, timeout=serial_timeout, **kwargs) as serial:
            with self.project.files.open('data', dest, mode=('a' if append else 'w')) as file:
                self.project.log('Starting streaming acquisition... ', end='')
                start_time = time()
                buffer = ''
                while True:
                    char = serial.read().decode('utf-8')
                    if len(char) < 1 or char not in alphabet:
                        continue
                    # when delimiter is found, check if it's a number
                    if char == delimiter or char == '\n':
                        try:
                            float(buffer)
                            file.write(buffer)
                            self.project.log('.', end='')
                            samples -= 1
                            if samples == 0:
                                break
                            else:
                                file.write(delimiter)
                        except ValueError:
                            self.project.log('ValueError', buffer)
                        buffer = ''
                    # append character to buffer
                    else:
                        buffer += char
                    # abort on timeout
                    if time() - start_time > timeout:
                        raise RuntimeError('Timeout')
                self.project.log('DONE')
        if dump:
            self.project.files.cat('data/%s' % dest)

