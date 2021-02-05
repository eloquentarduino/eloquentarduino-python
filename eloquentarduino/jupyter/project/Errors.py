class BoardNotFoundError(AssertionError):
    """
    Board pattern not found in the list of boards
    """
    pass


class MultipleBoardsFoundError(AssertionError):
    """
    Board search returned multiple matches
    """
    pass


class UploadNotVerifiedError(AssertionError):
    """
    Upload from arduino-cli is not verified
    """
    pass


class ArduinoCliCommandError(AssertionError):
    """
    The arduino-cli returned an error
    """
    pass


class NoSerialPortFoundError(AssertionError):
    """
    Cannot find any connected board
    """
    pass


class MultipleSerialPortsFoundError(AssertionError):
    """
    Found multiple serial ports, cannot determine which one to use
    """
    pass


class BadBoardResponseError(AssertionError):
    """
    The board replied with an unexpected format
    """
    pass


class BoardBenchmarkAlreadyExists(AssertionError):
    """
    The given board benchmark already exists
    """
    def __init__(self, key):
        message = 'A benchmark record for [%s] already exists' % str(key)
        super().__init__(message)