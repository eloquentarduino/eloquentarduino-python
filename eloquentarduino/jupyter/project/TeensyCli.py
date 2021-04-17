import os
import os.path
from platform import system
import re
from subprocess import STDOUT, CalledProcessError, check_output
from collections import namedtuple
import json
from time import time
from tempfile import TemporaryDirectory
from os import makedirs
import os.path
import hashlib


class TeensyCli:
    def run(self):
        """Run cli command and save output"""
        Mock = namedtuple('Mock', 'safe_output')
        folder = 'eloquent/BoardsBenchmarks'
        try:
            md5 = hashlib.md5()
            with open('/Users/simone/PycharmProjects/%s/sketches/tmp/tmp.ino' % folder) as sketch:
                sketch = sketch.read()
                if 'Classifier.h' in sketch:
                    with open('/Users/simone/PycharmProjects/%s/sketches/tmp/Classifier.h' % folder) as file:
                        md5.update(file.read().encode('utf-8'))
                else:
                    md5.update(sketch.encode('utf-8'))
            uid = md5.hexdigest()
            #print('digest', uid)
            build_dir = 'cache/build_%s' % uid
            cache_dir = 'cache/cache_%s' % uid
            makedirs(build_dir, 0o777, exist_ok=True)
            makedirs(cache_dir, 0o777, exist_ok=True)
            cpu_speed = 816
            cmd = '/Applications/Teensyduino.app/Contents/Java/arduino-builder -compile -logger=machine -hardware /Applications/Teensyduino.app/Contents/Java/hardware -hardware /Users/simone/Library/Arduino15/packages -hardware /Users/simone/Documents/Arduino/hardware -tools /Applications/Teensyduino.app/Contents/Java/tools-builder -tools /Applications/Teensyduino.app/Contents/Java/hardware/tools/avr -tools /Users/simone/Library/Arduino15/packages -built-in-libraries /Applications/Teensyduino.app/Contents/Java/libraries -libraries /Users/simone/Documents/Arduino/libraries -fqbn=teensy:avr:teensy40:usb=serial,speed=%d,opt=o3std,keys=en-us -ide-version=10813 -build-path %s -warnings=none -build-cache %s -verbose /Users/simone/PycharmProjects/%s/sketches/tmp/tmp.ino' % (cpu_speed, build_dir, cache_dir, folder)
            segments = cmd.split(' ')
            res = check_output(segments).decode('utf-8')
            lines = res.split('\n')
            lines = [l for l in lines if 'Sketch uses {0} bytes' in l or 'Global variables use' in l]
            flash = json.loads(lines[0].split('|||')[2].strip().replace(' ', ','))
            memory = json.loads(lines[1].split('|||')[2].strip().replace(' ', ','))
            output = 'Sketch uses %d bytes. Maximum is %d. Global variables use %d. Maximum is %d' % (flash[0], flash[1], memory[0], memory[1])
            #print('output', output)
            return Mock(safe_output=output)
            stop()
            self.project.logger.debug('(cwd %s) %s %s', self.cwd, self.executable, ' '.join([str(arg) for arg in self.arguments]))
            self.output = check_output([self.executable] + self.arguments, stderr=STDOUT, cwd=self.cwd).decode('utf-8')
            self.error = None
        except CalledProcessError as err:
            print('err', err)
            self.error = err.output.decode('utf-8')
            self.output = None