"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, cast

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Returns the product of x and y
    Args:
        x: float
        y: float

    Returns: float: the product of x and y

    """
    return x * y


def id(x: float) -> float:
    """Returns x
    Args:
        x: float

    Returns: float: x

    """
    return x


def add(x: float, y: float) -> float:
    """Returns the sum of two floats x and y
    Args:
        x: float
        y: float

    Returns: float: the sum of x and y

    """
    return x + y


def neg(x: float) -> float:
    """Returns a negation of float x
    Args:
        x: float

    Returns: float: the negation of x

    """
    return -x


def lt(x: float, y: float) -> float:
    """Returns 1.0 if x is less than y else 0.0
    Args:
        x: float
        y: float

    Returns: float: 1.0 if x is less than y else 0.0

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Returns 1.0 if x is equal to y else 0.0
    Args:
        x: float
        y: float

    Returns: float: 1.0 if x is equal to y else 0.0

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the maximum of two floats x and y
    Args:
        x: float
        y: float

    Returns: float: the maximum of x and y

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Returns 1.0 if x is close to y else 0.0
    Args:
        x: float
        y: float

    Returns: float: 1.0 if x is close to y else 0.0

    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Returns the sigmoid of x
    Args:
        x: float

    Returns: float: the sigmoid of x

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Returns the relu of x
    Args:
        x: float

    Returns: float: the relu of x

    """
    return x if x >= 0 else 0.0


def log(x: float) -> float:
    """Returns the log of x
    Args:
        x: float

    Returns: float: the log of x

    """
    return math.log(x)


def exp(x: float) -> float:
    """Returns the exp of x
    Args:
        x: float

    Returns: float: the exp of x

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Returns the inverse of x
    Args:
        x: float

    Returns: float: the inverse of x

    """
    if x == 0.0:
        return float("inf")
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Returns the derivative of the log of x times a second argument
        x: float
        d: float

    Returns: float: the derivative of the log of x times a second argument

    """
    return d / x


def inv_back(x: float, d: float) -> float:
    """Returns the derivative of the inverse of x times a second argument
    Args:
        x: float
        d: float

    Returns: float: the derivative of the inverse of x times a second argument

    """
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Returns the derivative of ReLU of x times a second argument
    Args:
        x: float
        d: float

    Returns: float: derivative of ReLU of x times a second argument

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(f: Callable[[object], object], xs: Iterable[object]) -> Iterable[object]:
    """A higher-order function that applies a given function to each element of an iterable
    Args:
        f: function to apply
        xs: iterable to apply function to

    Returns: iterable of the results of applying the function

    """
    for x in xs:
        yield f(x)


def zipWith(
    f: Callable[[object, object], object], xs: Iterable[object], ys: Iterable[object]
) -> Iterable[object]:
    """A higher-order function that combines elements from two iterables of the same size using a given function
    Args:
        f: function to apply to pairs of elements
        xs: first iterable
        ys: second iterable, assumed to be the same size as xs

    Returns: iterable of the results of applying the function to pairs of elements

    """
    it = iter(ys)
    for x in xs:
        y = next(it)
        yield f(x, y)


def reduce(f: Callable[[object, object], object], xs: Iterable[object]) -> object:
    """A higher-order function that reduces an iterable to a single value using a given function
    Args:
        f: function to apply to pairs of elements
        xs: iterable to apply function to

    Returns: single value of the results of applying the function to each pair of elements

    """
    it = iter(xs)
    val = next(it)
    for x in it:
        val = f(val, x)
    return val


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map
    Args:
        xs: list of floats

    Returns: list of negated floats

    """
    return cast(Iterable[float], map(cast(Callable[[object], object], neg), xs))


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith
    Args:
        xs: first list of floats
        ys: second list of floats, assumed to be the same size as xs

    Returns: list of sums of corresponding elements

    """
    return cast(
        Iterable[float], zipWith(cast(Callable[[object, object], object], add), xs, ys)
    )


def sum(xs: Iterable[float]) -> float:
    """Sum all elements in a list using reduce
    Args:
        xs: list of floats

    Returns: sum of all elements

    """
    if not xs:
        return 0.0
    return cast(float, reduce(cast(Callable[[object, object], object], add), xs))


def prod(xs: Iterable[float]) -> float:
    """Product all elements in a list using reduce
    Args:
        xs: list of floats

    Returns: product of all elements

    """
    return cast(float, reduce(cast(Callable[[object, object], object], mul), xs))
