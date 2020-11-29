class BoardNotFoundError(AssertionError):
    """
    Board pattern not found in the list of boards
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