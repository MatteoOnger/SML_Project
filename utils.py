from typing import TypeVar


T = TypeVar("T")


def round_wrp(x :T, dig :int) -> T:
    """
    Wrap the round function: it rounds ``x`` to the decimal place only if it is a number,
    otherwise it returns ``x`` as is.

    Parameters
    ----------
    x : T
        Input possibly to be rounded.
    dig : int
        Number of decimal places to keep.

    Returns
    -------
    :T
        ``x`` possibly rounded.
    """
    return round(x, dig) if isinstance(x, (int, float, complex)) else x