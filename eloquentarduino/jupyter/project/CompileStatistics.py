import re


class CompileStatistics:
    """Parse arduino-cli compilation statistics"""
    def __init__(self, compile_output):
        pattern = r"Sketch uses (\d+) bytes.+?Maximum is (\d+).+?Global variables use (\d+).+?Maximum is (\d+)"
        match = re.search(pattern, compile_output.replace("\n", ""))
        self.flash = 0
        self.flash_max = 0
        self.memory = 0
        self.memory_max = 0
        if match is not None:
            self.flash, self.flash_max, self.memory, self.memory_max = [int(g) for g in match.groups()]

    @property
    def flash_percent(self):
        """Get flash percent"""
        return int(100 * self.flash / self.flash_max) if self.flash_max > 0 else 0

    @property
    def memory_percent(self):
        """Get memory percent"""
        return int(100 * self.memory / self.memory_max) if self.memory_max > 0 else 0