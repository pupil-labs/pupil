class Timestamp(int):

    def __new__(cls, *args, nanoseconds: int = None, seconds: float = None, **kwargs):
        if nanoseconds is not None and seconds is None:
            nanoseconds = int(nanoseconds)
        elif nanoseconds is None and seconds is not None:
            nanoseconds = int(seconds * 1000)
        else:
            raise ValueError(f"Must provide EITHER `nanoseconds` ({nanoseconds}) OR `seconds` ({seconds}) argument.")
        return  super(cls, cls).__new__(cls, nanoseconds)

    @property
    def raw_nanoseconds(self) -> int:
        return int(self)

    @property
    def raw_seconds(self) -> float:
        return float(self.raw_nanoseconds / 1000)

    def __add__(self, other):
        if not isinstance(other, Timestamp):
            error_msg = "Addition only supported between `Timestamp` instances. "
            error_msg += "Please use `Timestamp(nanoseconds=)` or `Timestamp(seconds=)` "
            error_msg += "initializers to explicitly create a `Timestamp` instance."
            raise ValueError(error_msg)
        nanoseconds = self.raw_nanoseconds + other.raw_nanoseconds
        return self.__class__(nanoseconds=nanoseconds)

    def __sub__(self, other):
        if not isinstance(other, Timestamp):
            error_msg = "Subtraction only supported between `Timestamp` instances. "
            error_msg += "Please use `Timestamp(nanoseconds=)` or `Timestamp(seconds=)` "
            error_msg += "initializers to explicitly create a `Timestamp` instance."
            raise ValueError(error_msg)
        nanoseconds = self.raw_nanoseconds - other.raw_nanoseconds
        return self.__class__(nanoseconds=nanoseconds)

    def __mul__(self, other):
        raise NotImplementedError("Timestamp multiplication not supported")

    def __div__(self, other):
        raise NotImplementedError("Timestamp division not supported")

    def __str__(self):
        return self.raw_nanoseconds.__str__()

    def __repr__(self):
        return f"{type(self).__name__}({self.raw_nanoseconds})"
