
class BufferSizeException(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "Buffer size is too low to get a mini-batch."
        super().__init__(message)


class NodeNotFoundException(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "The chosen node was not found."
        super().__init__(message)