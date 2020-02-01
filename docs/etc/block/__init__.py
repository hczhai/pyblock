
"""Python3 wrapper for block 1.5.3 (spin adapted)."""


def load_rotation_matrix(*args, **kwargs):
    """load_rotation_matrix(arg0: block.VectorInt, arg1: block.VectorMatrix, arg2: int) -> None

Load rotation matrix."""
    pass


def save_rotation_matrix(*args, **kwargs):
    """save_rotation_matrix(arg0: block.VectorInt, arg1: block.VectorMatrix, arg2: int) -> None

Save rotation matrix."""
    pass


class DiagonalMatrix:
    """NEWMAT10 diagonal matrix."""

    @property
    def ref(self):
        """A numpy.ndarray reference."""
        pass

    @property
    def rows(self):
        pass

    @property
    def cols(self):
        pass

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.DiagonalMatrix) -> None

2. __init__(self: block.DiagonalMatrix, arg0: numpy.ndarray[float64[m, n], flags.writeable, flags.c_contiguous]) -> None"""
        pass

    def __add__(self, *args, **kwargs):
        """__add__(self: block.DiagonalMatrix, arg0: block.DiagonalMatrix) -> block.DiagonalMatrix"""
        pass

    def resize(self, *args, **kwargs):
        """resize(self: block.DiagonalMatrix, nr: int) -> None"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.DiagonalMatrix) -> str"""
        pass


class Matrix:
    """NEWMAT10 matrix."""

    @property
    def ref(self):
        """A numpy.ndarray reference."""
        pass

    @property
    def rows(self):
        pass

    @property
    def cols(self):
        pass

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.Matrix) -> None

2. __init__(self: block.Matrix, arg0: numpy.ndarray[float64[m, n], flags.writeable, flags.c_contiguous]) -> None"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.Matrix) -> str"""
        pass


class VectorBool:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.VectorBool) -> None

2. __init__(self: block.VectorBool, arg0: block.VectorBool) -> None

Copy constructor

3. __init__(self: block.VectorBool, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.VectorBool, arg0: block.VectorBool) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.VectorBool, arg0: block.VectorBool) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.VectorBool, x: bool) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.VectorBool, x: bool) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.VectorBool, x: bool) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.VectorBool) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.VectorBool, x: bool) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.VectorBool, L: block.VectorBool) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.VectorBool, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.VectorBool, i: int, x: bool) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.VectorBool) -> bool

Remove and return the last item

2. pop(self: block.VectorBool, i: int) -> bool

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.VectorBool, arg0: int, arg1: bool) -> None

2. __setitem__(self: block.VectorBool, arg0: slice, arg1: block.VectorBool) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.VectorBool, s: slice) -> block.VectorBool

Retrieve list elements using a slice object

2. __getitem__(self: block.VectorBool, arg0: int) -> bool"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.VectorBool, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.VectorBool, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.VectorBool) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.VectorBool) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.VectorBool) -> int"""
        pass


class VectorDouble:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.VectorDouble) -> None

2. __init__(self: block.VectorDouble, arg0: block.VectorDouble) -> None

Copy constructor

3. __init__(self: block.VectorDouble, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.VectorDouble, arg0: block.VectorDouble) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.VectorDouble, arg0: block.VectorDouble) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.VectorDouble, x: float) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.VectorDouble, x: float) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.VectorDouble, x: float) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.VectorDouble) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.VectorDouble, x: float) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.VectorDouble, L: block.VectorDouble) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.VectorDouble, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.VectorDouble, i: int, x: float) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.VectorDouble) -> float

Remove and return the last item

2. pop(self: block.VectorDouble, i: int) -> float

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.VectorDouble, arg0: int, arg1: float) -> None

2. __setitem__(self: block.VectorDouble, arg0: slice, arg1: block.VectorDouble) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.VectorDouble, s: slice) -> block.VectorDouble

Retrieve list elements using a slice object

2. __getitem__(self: block.VectorDouble, arg0: int) -> float"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.VectorDouble, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.VectorDouble, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.VectorDouble) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.VectorDouble) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.VectorDouble) -> int"""
        pass


class VectorInt:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.VectorInt) -> None

2. __init__(self: block.VectorInt, arg0: block.VectorInt) -> None

Copy constructor

3. __init__(self: block.VectorInt, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.VectorInt, arg0: block.VectorInt) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.VectorInt, arg0: block.VectorInt) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.VectorInt, x: int) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.VectorInt, x: int) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.VectorInt, x: int) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.VectorInt) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.VectorInt, x: int) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.VectorInt, L: block.VectorInt) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.VectorInt, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.VectorInt, i: int, x: int) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.VectorInt) -> int

Remove and return the last item

2. pop(self: block.VectorInt, i: int) -> int

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.VectorInt, arg0: int, arg1: int) -> None

2. __setitem__(self: block.VectorInt, arg0: slice, arg1: block.VectorInt) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.VectorInt, s: slice) -> block.VectorInt

Retrieve list elements using a slice object

2. __getitem__(self: block.VectorInt, arg0: int) -> int"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.VectorInt, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.VectorInt, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.VectorInt) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.VectorInt) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.VectorInt) -> int"""
        pass


class VectorMatrix:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.VectorMatrix) -> None

2. __init__(self: block.VectorMatrix, arg0: block.VectorMatrix) -> None

Copy constructor

3. __init__(self: block.VectorMatrix, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.VectorMatrix, arg0: block.VectorMatrix) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.VectorMatrix, arg0: block.VectorMatrix) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.VectorMatrix, x: block.Matrix) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.VectorMatrix, x: block.Matrix) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.VectorMatrix, x: block.Matrix) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.VectorMatrix) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.VectorMatrix, x: block.Matrix) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.VectorMatrix, L: block.VectorMatrix) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.VectorMatrix, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.VectorMatrix, i: int, x: block.Matrix) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.VectorMatrix) -> block.Matrix

Remove and return the last item

2. pop(self: block.VectorMatrix, i: int) -> block.Matrix

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.VectorMatrix, arg0: int, arg1: block.Matrix) -> None

2. __setitem__(self: block.VectorMatrix, arg0: slice, arg1: block.VectorMatrix) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.VectorMatrix, s: slice) -> block.VectorMatrix

Retrieve list elements using a slice object

2. __getitem__(self: block.VectorMatrix, arg0: int) -> block.Matrix"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.VectorMatrix, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.VectorMatrix, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.VectorMatrix) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.VectorMatrix) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.VectorMatrix) -> int"""
        pass


class VectorVectorInt:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.VectorVectorInt) -> None

2. __init__(self: block.VectorVectorInt, arg0: block.VectorVectorInt) -> None

Copy constructor

3. __init__(self: block.VectorVectorInt, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.VectorVectorInt, arg0: block.VectorVectorInt) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.VectorVectorInt, arg0: block.VectorVectorInt) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.VectorVectorInt, x: block.VectorInt) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.VectorVectorInt, x: block.VectorInt) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.VectorVectorInt, x: block.VectorInt) -> bool

Return true the container contains ``x``"""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.VectorVectorInt, x: block.VectorInt) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.VectorVectorInt, L: block.VectorVectorInt) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.VectorVectorInt, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.VectorVectorInt, i: int, x: block.VectorInt) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.VectorVectorInt) -> block.VectorInt

Remove and return the last item

2. pop(self: block.VectorVectorInt, i: int) -> block.VectorInt

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.VectorVectorInt, arg0: int, arg1: block.VectorInt) -> None

2. __setitem__(self: block.VectorVectorInt, arg0: slice, arg1: block.VectorVectorInt) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.VectorVectorInt, s: slice) -> block.VectorVectorInt

Retrieve list elements using a slice object

2. __getitem__(self: block.VectorVectorInt, arg0: int) -> block.VectorInt"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.VectorVectorInt, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.VectorVectorInt, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.VectorVectorInt) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.VectorVectorInt) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.VectorVectorInt) -> int"""
        pass

