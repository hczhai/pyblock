
"""Classes for operator matrices and operations."""

import enum

def multiply_with_own_transpose(*args, **kwargs):
    """multiply_with_own_transpose(arg0: block.operator.StackSparseMatrix, arg1: block.operator.StackSparseMatrix, arg2: float) -> None"""
    pass


class MapPairInt:

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.MapPairInt) -> None"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.MapPairInt) -> bool

Check whether the map is nonempty"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.MapPairInt) -> iterator"""
        pass

    def items(self, *args, **kwargs):
        """items(self: block.operator.MapPairInt) -> iterator"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self: block.operator.MapPairInt, arg0: Tuple[int, int]) -> int"""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.MapPairInt, arg0: Tuple[int, int]) -> bool"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self: block.operator.MapPairInt, arg0: Tuple[int, int], arg1: int) -> None"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(self: block.operator.MapPairInt, arg0: Tuple[int, int]) -> None"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.MapPairInt) -> int"""
        pass


class OpTypes(enum.Enum):
    """Types of operators (enumerator).

Members:

  Hamiltonian

  Cre

  CreCre

  DesDesComp

  CreDes

  CreDesComp

  CreCreDesComp

  Des

  DesDes

  CreCreComp

  DesCre

  DesCreComp

  CreDesDesComp

  Overlap"""
    Hamiltonian = enum.auto()
    Cre = enum.auto()
    CreCre = enum.auto()
    DesDesComp = enum.auto()
    CreDes = enum.auto()
    CreDesComp = enum.auto()
    CreCreDesComp = enum.auto()
    Des = enum.auto()
    DesDes = enum.auto()
    CreCreComp = enum.auto()
    DesCre = enum.auto()
    DesCreComp = enum.auto()
    CreDesDesComp = enum.auto()
    Overlap = enum.auto()


class OperatorArrayBase:
    pass


class OperatorArrayCre(OperatorArrayBase):
    """An array of Cre operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayCre) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayCre, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayCre, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayCre, arg0: int) -> bool

Query whether the element is non-zero (in local or global storage). The parameters are site indices."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayCre, arg0: int) -> bool

Query whether the element is non-zero in local storage. The parameters are site indices."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayCre, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given site indices (in local storage)."""
        pass


class OperatorArrayCreCre(OperatorArrayBase):
    """An array of CreCre operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayCreCre) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayCreCre, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayCreCre, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayCreCre, arg0: int, arg1: int) -> bool

Query whether the element is non-zero (in local or global storage). The parameters are site indices."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayCreCre, arg0: int, arg1: int) -> bool

Query whether the element is non-zero in local storage. The parameters are site indices."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayCreCre, arg0: int, arg1: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given site indices (in local storage)."""
        pass


class OperatorArrayCreCreComp(OperatorArrayBase):
    """An array of CreCreComp operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayCreCreComp) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayCreCreComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayCreCreComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayCreCreComp, arg0: int, arg1: int) -> bool

Query whether the element is non-zero (in local or global storage). The parameters are site indices."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayCreCreComp, arg0: int, arg1: int) -> bool

Query whether the element is non-zero in local storage. The parameters are site indices."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayCreCreComp, arg0: int, arg1: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given site indices (in local storage)."""
        pass


class OperatorArrayCreCreDesComp(OperatorArrayBase):
    """An array of CreCreDesComp operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayCreCreDesComp) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayCreCreDesComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayCreCreDesComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayCreCreDesComp, arg0: int) -> bool

Query whether the element is non-zero (in local or global storage). The parameters are site indices."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayCreCreDesComp, arg0: int) -> bool

Query whether the element is non-zero in local storage. The parameters are site indices."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayCreCreDesComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given site indices (in local storage)."""
        pass


class OperatorArrayCreDes(OperatorArrayBase):
    """An array of CreDes operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayCreDes) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayCreDes, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayCreDes, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayCreDes, arg0: int, arg1: int) -> bool

Query whether the element is non-zero (in local or global storage). The parameters are site indices."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayCreDes, arg0: int, arg1: int) -> bool

Query whether the element is non-zero in local storage. The parameters are site indices."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayCreDes, arg0: int, arg1: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given site indices (in local storage)."""
        pass


class OperatorArrayCreDesComp(OperatorArrayBase):
    """An array of CreDesComp operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayCreDesComp) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayCreDesComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayCreDesComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayCreDesComp, arg0: int, arg1: int) -> bool

Query whether the element is non-zero (in local or global storage). The parameters are site indices."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayCreDesComp, arg0: int, arg1: int) -> bool

Query whether the element is non-zero in local storage. The parameters are site indices."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayCreDesComp, arg0: int, arg1: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given site indices (in local storage)."""
        pass


class OperatorArrayCreDesDesComp(OperatorArrayBase):
    """An array of CreDesDesComp operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayCreDesDesComp) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayCreDesDesComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayCreDesDesComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayCreDesDesComp, arg0: int) -> bool

Query whether the element is non-zero (in local or global storage). The parameters are site indices."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayCreDesDesComp, arg0: int) -> bool

Query whether the element is non-zero in local storage. The parameters are site indices."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayCreDesDesComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given site indices (in local storage)."""
        pass


class OperatorArrayDes(OperatorArrayBase):
    """An array of Des operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayDes) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayDes, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayDes, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayDes, arg0: int) -> bool

Query whether the element is non-zero (in local or global storage). The parameters are site indices."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayDes, arg0: int) -> bool

Query whether the element is non-zero in local storage. The parameters are site indices."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayDes, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given site indices (in local storage)."""
        pass


class OperatorArrayDesCre(OperatorArrayBase):
    """An array of DesCre operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayDesCre) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayDesCre, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayDesCre, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayDesCre, arg0: int, arg1: int) -> bool

Query whether the element is non-zero (in local or global storage). The parameters are site indices."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayDesCre, arg0: int, arg1: int) -> bool

Query whether the element is non-zero in local storage. The parameters are site indices."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayDesCre, arg0: int, arg1: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given site indices (in local storage)."""
        pass


class OperatorArrayDesCreComp(OperatorArrayBase):
    """An array of DesCreComp operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayDesCreComp) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayDesCreComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayDesCreComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayDesCreComp, arg0: int, arg1: int) -> bool

Query whether the element is non-zero (in local or global storage). The parameters are site indices."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayDesCreComp, arg0: int, arg1: int) -> bool

Query whether the element is non-zero in local storage. The parameters are site indices."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayDesCreComp, arg0: int, arg1: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given site indices (in local storage)."""
        pass


class OperatorArrayDesDes(OperatorArrayBase):
    """An array of DesDes operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayDesDes) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayDesDes, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayDesDes, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayDesDes, arg0: int, arg1: int) -> bool

Query whether the element is non-zero (in local or global storage). The parameters are site indices."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayDesDes, arg0: int, arg1: int) -> bool

Query whether the element is non-zero in local storage. The parameters are site indices."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayDesDes, arg0: int, arg1: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given site indices (in local storage)."""
        pass


class OperatorArrayDesDesComp(OperatorArrayBase):
    """An array of DesDesComp operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayDesDesComp) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayDesDesComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayDesDesComp, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayDesDesComp, arg0: int, arg1: int) -> bool

Query whether the element is non-zero (in local or global storage). The parameters are site indices."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayDesDesComp, arg0: int, arg1: int) -> bool

Query whether the element is non-zero in local storage. The parameters are site indices."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayDesDesComp, arg0: int, arg1: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given site indices (in local storage)."""
        pass


class OperatorArrayHamiltonian(OperatorArrayBase):
    """An array of Hamiltonian operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayHamiltonian) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayHamiltonian, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayHamiltonian, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayHamiltonian) -> bool

Query whether the element is non-zero (in local or global storage)."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayHamiltonian) -> bool

Query whether the element is non-zero in local storage."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayHamiltonian) -> block.operator.VectorStackSparseMatrix

Get the array of operators (for different spin quantum numbers, in local storage)."""
        pass


class OperatorArrayOverlap(OperatorArrayBase):
    """An array of Overlap operators defined at different sites."""

    @property
    def op_string(self):
        """Name of the type of operators contained in this array."""
        pass

    @property
    def n_local_nz(self):
        """Number of non-zero elements in local storage."""
        pass

    @property
    def n_global_nz(self):
        """Number of non-zero elements in global storage."""
        pass

    @property
    def local_indices(self):
        """A 2d array contains the site indices of non-zero elements in local storage. It gives a map from flattened single index to multiple site indices (which is represented as an array)."""
        pass

    @property
    def global_indices(self):
        """A 1d array contains the site indices of non-zero elements (in local or global storage). It gives a map from flattened single index to multiple site indices. Then this array itself is flattened."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorArrayOverlap) -> None"""
        pass

    def local_element_linear(self, *args, **kwargs):
        """local_element_linear(self: block.operator.OperatorArrayOverlap, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in local storage)."""
        pass

    def global_element_linear(self, *args, **kwargs):
        """global_element_linear(self: block.operator.OperatorArrayOverlap, arg0: int) -> block.operator.VectorStackSparseMatrix

Get an array of operators (for different spin quantum numbers) defined for the given (flattened) linear index (in global storage)."""
        pass

    def has_global(self, *args, **kwargs):
        """has_global(self: block.operator.OperatorArrayOverlap) -> bool

Query whether the element is non-zero (in local or global storage)."""
        pass

    def has_local(self, *args, **kwargs):
        """has_local(self: block.operator.OperatorArrayOverlap) -> bool

Query whether the element is non-zero in local storage."""
        pass

    def local_element(self, *args, **kwargs):
        """local_element(self: block.operator.OperatorArrayOverlap) -> block.operator.VectorStackSparseMatrix

Get the array of operators (for different spin quantum numbers, in local storage)."""
        pass


class StackMatrix:
    """Very simple Matrix class that provides a Matrix type interface for a double array. It does not own its own data.

Note that the C++ class used indices counting from 1. Here we count from 0. Row-major (C) storage."""

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

1. __init__(self: block.operator.StackMatrix) -> None

2. __init__(self: block.operator.StackMatrix, arg0: numpy.ndarray[float64[m, n], flags.writeable, flags.c_contiguous]) -> None"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.StackMatrix) -> str"""
        pass


class StackSparseMatrix:
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    @property
    def total_memory(self):
        pass

    @property
    def non_zero_blocks(self):
        """A list of non zero blocks. Each element in the list is a pair of a pair of bra and ket indices, and :class:`StackMatrix`."""
        pass

    @property
    def map_to_non_zero_blocks(self):
        """A map from pair of bra and ket indices, to the index in :attr:`StackSparseMatrix.non_zero_blocks`."""
        pass

    @property
    def delta_quantum(self):
        """Allowed change of quantum numbers between states."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.StackSparseMatrix) -> None"""
        pass

    def clear(self, *args, **kwargs):
        """clear(self: block.operator.StackSparseMatrix) -> None"""
        pass

    def allocate(self, *args, **kwargs):
        """allocate(self: block.operator.StackSparseMatrix, arg0: block.symmetry.StateInfo) -> None"""
        pass

    def deallocate(self, *args, **kwargs):
        """deallocate(self: block.operator.StackSparseMatrix) -> None"""
        pass


class DensityMatrix(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.DensityMatrix) -> None"""
        pass


class OperatorCre(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorCre) -> None"""
        pass


class OperatorCreCre(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorCreCre) -> None"""
        pass


class OperatorCreCreComp(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorCreCreComp) -> None"""
        pass


class OperatorCreCreDesComp(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorCreCreDesComp) -> None"""
        pass


class OperatorCreDes(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorCreDes) -> None"""
        pass


class OperatorCreDesComp(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorCreDesComp) -> None"""
        pass


class OperatorCreDesDesComp(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorCreDesDesComp) -> None"""
        pass


class OperatorDes(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorDes) -> None"""
        pass


class OperatorDesCre(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorDesCre) -> None"""
        pass


class OperatorDesCreComp(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorDesCreComp) -> None"""
        pass


class OperatorDesDes(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorDesDes) -> None"""
        pass


class OperatorDesDesComp(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorDesDesComp) -> None"""
        pass


class OperatorHamiltonian(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorHamiltonian) -> None"""
        pass


class OperatorOverlap(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.OperatorOverlap) -> None"""
        pass


class VectorCre:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorCre) -> None

2. __init__(self: block.operator.VectorCre, arg0: block.operator.VectorCre) -> None

Copy constructor

3. __init__(self: block.operator.VectorCre, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorCre, arg0: block.operator.VectorCre) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorCre, arg0: block.operator.VectorCre) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorCre, x: block.operator.OperatorCre) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorCre, x: block.operator.OperatorCre) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorCre, x: block.operator.OperatorCre) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorCre) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorCre, x: block.operator.OperatorCre) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorCre, L: block.operator.VectorCre) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorCre, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorCre, i: int, x: block.operator.OperatorCre) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorCre) -> block.operator.OperatorCre

Remove and return the last item

2. pop(self: block.operator.VectorCre, i: int) -> block.operator.OperatorCre

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorCre, arg0: int, arg1: block.operator.OperatorCre) -> None

2. __setitem__(self: block.operator.VectorCre, arg0: slice, arg1: block.operator.VectorCre) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorCre, s: slice) -> block.operator.VectorCre

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorCre, arg0: int) -> block.operator.OperatorCre"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorCre, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorCre, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorCre) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorCre) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorCre) -> int"""
        pass


class VectorCreCre:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorCreCre) -> None

2. __init__(self: block.operator.VectorCreCre, arg0: block.operator.VectorCreCre) -> None

Copy constructor

3. __init__(self: block.operator.VectorCreCre, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorCreCre, arg0: block.operator.VectorCreCre) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorCreCre, arg0: block.operator.VectorCreCre) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorCreCre, x: block.operator.OperatorCreCre) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorCreCre, x: block.operator.OperatorCreCre) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorCreCre, x: block.operator.OperatorCreCre) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorCreCre) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorCreCre, x: block.operator.OperatorCreCre) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorCreCre, L: block.operator.VectorCreCre) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorCreCre, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorCreCre, i: int, x: block.operator.OperatorCreCre) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorCreCre) -> block.operator.OperatorCreCre

Remove and return the last item

2. pop(self: block.operator.VectorCreCre, i: int) -> block.operator.OperatorCreCre

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorCreCre, arg0: int, arg1: block.operator.OperatorCreCre) -> None

2. __setitem__(self: block.operator.VectorCreCre, arg0: slice, arg1: block.operator.VectorCreCre) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorCreCre, s: slice) -> block.operator.VectorCreCre

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorCreCre, arg0: int) -> block.operator.OperatorCreCre"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorCreCre, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorCreCre, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorCreCre) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorCreCre) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorCreCre) -> int"""
        pass


class VectorCreCreComp:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorCreCreComp) -> None

2. __init__(self: block.operator.VectorCreCreComp, arg0: block.operator.VectorCreCreComp) -> None

Copy constructor

3. __init__(self: block.operator.VectorCreCreComp, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorCreCreComp, arg0: block.operator.VectorCreCreComp) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorCreCreComp, arg0: block.operator.VectorCreCreComp) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorCreCreComp, x: block.operator.OperatorCreCreComp) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorCreCreComp, x: block.operator.OperatorCreCreComp) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorCreCreComp, x: block.operator.OperatorCreCreComp) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorCreCreComp) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorCreCreComp, x: block.operator.OperatorCreCreComp) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorCreCreComp, L: block.operator.VectorCreCreComp) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorCreCreComp, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorCreCreComp, i: int, x: block.operator.OperatorCreCreComp) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorCreCreComp) -> block.operator.OperatorCreCreComp

Remove and return the last item

2. pop(self: block.operator.VectorCreCreComp, i: int) -> block.operator.OperatorCreCreComp

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorCreCreComp, arg0: int, arg1: block.operator.OperatorCreCreComp) -> None

2. __setitem__(self: block.operator.VectorCreCreComp, arg0: slice, arg1: block.operator.VectorCreCreComp) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorCreCreComp, s: slice) -> block.operator.VectorCreCreComp

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorCreCreComp, arg0: int) -> block.operator.OperatorCreCreComp"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorCreCreComp, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorCreCreComp, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorCreCreComp) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorCreCreComp) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorCreCreComp) -> int"""
        pass


class VectorCreCreDesComp:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorCreCreDesComp) -> None

2. __init__(self: block.operator.VectorCreCreDesComp, arg0: block.operator.VectorCreCreDesComp) -> None

Copy constructor

3. __init__(self: block.operator.VectorCreCreDesComp, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorCreCreDesComp, arg0: block.operator.VectorCreCreDesComp) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorCreCreDesComp, arg0: block.operator.VectorCreCreDesComp) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorCreCreDesComp, x: block.operator.OperatorCreCreDesComp) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorCreCreDesComp, x: block.operator.OperatorCreCreDesComp) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorCreCreDesComp, x: block.operator.OperatorCreCreDesComp) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorCreCreDesComp) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorCreCreDesComp, x: block.operator.OperatorCreCreDesComp) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorCreCreDesComp, L: block.operator.VectorCreCreDesComp) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorCreCreDesComp, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorCreCreDesComp, i: int, x: block.operator.OperatorCreCreDesComp) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorCreCreDesComp) -> block.operator.OperatorCreCreDesComp

Remove and return the last item

2. pop(self: block.operator.VectorCreCreDesComp, i: int) -> block.operator.OperatorCreCreDesComp

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorCreCreDesComp, arg0: int, arg1: block.operator.OperatorCreCreDesComp) -> None

2. __setitem__(self: block.operator.VectorCreCreDesComp, arg0: slice, arg1: block.operator.VectorCreCreDesComp) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorCreCreDesComp, s: slice) -> block.operator.VectorCreCreDesComp

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorCreCreDesComp, arg0: int) -> block.operator.OperatorCreCreDesComp"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorCreCreDesComp, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorCreCreDesComp, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorCreCreDesComp) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorCreCreDesComp) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorCreCreDesComp) -> int"""
        pass


class VectorCreDes:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorCreDes) -> None

2. __init__(self: block.operator.VectorCreDes, arg0: block.operator.VectorCreDes) -> None

Copy constructor

3. __init__(self: block.operator.VectorCreDes, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorCreDes, arg0: block.operator.VectorCreDes) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorCreDes, arg0: block.operator.VectorCreDes) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorCreDes, x: block.operator.OperatorCreDes) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorCreDes, x: block.operator.OperatorCreDes) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorCreDes, x: block.operator.OperatorCreDes) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorCreDes) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorCreDes, x: block.operator.OperatorCreDes) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorCreDes, L: block.operator.VectorCreDes) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorCreDes, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorCreDes, i: int, x: block.operator.OperatorCreDes) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorCreDes) -> block.operator.OperatorCreDes

Remove and return the last item

2. pop(self: block.operator.VectorCreDes, i: int) -> block.operator.OperatorCreDes

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorCreDes, arg0: int, arg1: block.operator.OperatorCreDes) -> None

2. __setitem__(self: block.operator.VectorCreDes, arg0: slice, arg1: block.operator.VectorCreDes) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorCreDes, s: slice) -> block.operator.VectorCreDes

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorCreDes, arg0: int) -> block.operator.OperatorCreDes"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorCreDes, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorCreDes, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorCreDes) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorCreDes) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorCreDes) -> int"""
        pass


class VectorCreDesComp:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorCreDesComp) -> None

2. __init__(self: block.operator.VectorCreDesComp, arg0: block.operator.VectorCreDesComp) -> None

Copy constructor

3. __init__(self: block.operator.VectorCreDesComp, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorCreDesComp, arg0: block.operator.VectorCreDesComp) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorCreDesComp, arg0: block.operator.VectorCreDesComp) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorCreDesComp, x: block.operator.OperatorCreDesComp) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorCreDesComp, x: block.operator.OperatorCreDesComp) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorCreDesComp, x: block.operator.OperatorCreDesComp) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorCreDesComp) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorCreDesComp, x: block.operator.OperatorCreDesComp) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorCreDesComp, L: block.operator.VectorCreDesComp) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorCreDesComp, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorCreDesComp, i: int, x: block.operator.OperatorCreDesComp) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorCreDesComp) -> block.operator.OperatorCreDesComp

Remove and return the last item

2. pop(self: block.operator.VectorCreDesComp, i: int) -> block.operator.OperatorCreDesComp

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorCreDesComp, arg0: int, arg1: block.operator.OperatorCreDesComp) -> None

2. __setitem__(self: block.operator.VectorCreDesComp, arg0: slice, arg1: block.operator.VectorCreDesComp) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorCreDesComp, s: slice) -> block.operator.VectorCreDesComp

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorCreDesComp, arg0: int) -> block.operator.OperatorCreDesComp"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorCreDesComp, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorCreDesComp, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorCreDesComp) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorCreDesComp) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorCreDesComp) -> int"""
        pass


class VectorCreDesDesComp:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorCreDesDesComp) -> None

2. __init__(self: block.operator.VectorCreDesDesComp, arg0: block.operator.VectorCreDesDesComp) -> None

Copy constructor

3. __init__(self: block.operator.VectorCreDesDesComp, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorCreDesDesComp, arg0: block.operator.VectorCreDesDesComp) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorCreDesDesComp, arg0: block.operator.VectorCreDesDesComp) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorCreDesDesComp, x: block.operator.OperatorCreDesDesComp) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorCreDesDesComp, x: block.operator.OperatorCreDesDesComp) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorCreDesDesComp, x: block.operator.OperatorCreDesDesComp) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorCreDesDesComp) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorCreDesDesComp, x: block.operator.OperatorCreDesDesComp) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorCreDesDesComp, L: block.operator.VectorCreDesDesComp) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorCreDesDesComp, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorCreDesDesComp, i: int, x: block.operator.OperatorCreDesDesComp) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorCreDesDesComp) -> block.operator.OperatorCreDesDesComp

Remove and return the last item

2. pop(self: block.operator.VectorCreDesDesComp, i: int) -> block.operator.OperatorCreDesDesComp

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorCreDesDesComp, arg0: int, arg1: block.operator.OperatorCreDesDesComp) -> None

2. __setitem__(self: block.operator.VectorCreDesDesComp, arg0: slice, arg1: block.operator.VectorCreDesDesComp) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorCreDesDesComp, s: slice) -> block.operator.VectorCreDesDesComp

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorCreDesDesComp, arg0: int) -> block.operator.OperatorCreDesDesComp"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorCreDesDesComp, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorCreDesDesComp, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorCreDesDesComp) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorCreDesDesComp) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorCreDesDesComp) -> int"""
        pass


class VectorDes:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorDes) -> None

2. __init__(self: block.operator.VectorDes, arg0: block.operator.VectorDes) -> None

Copy constructor

3. __init__(self: block.operator.VectorDes, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorDes, arg0: block.operator.VectorDes) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorDes, arg0: block.operator.VectorDes) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorDes, x: block.operator.OperatorDes) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorDes, x: block.operator.OperatorDes) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorDes, x: block.operator.OperatorDes) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorDes) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorDes, x: block.operator.OperatorDes) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorDes, L: block.operator.VectorDes) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorDes, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorDes, i: int, x: block.operator.OperatorDes) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorDes) -> block.operator.OperatorDes

Remove and return the last item

2. pop(self: block.operator.VectorDes, i: int) -> block.operator.OperatorDes

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorDes, arg0: int, arg1: block.operator.OperatorDes) -> None

2. __setitem__(self: block.operator.VectorDes, arg0: slice, arg1: block.operator.VectorDes) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorDes, s: slice) -> block.operator.VectorDes

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorDes, arg0: int) -> block.operator.OperatorDes"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorDes, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorDes, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorDes) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorDes) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorDes) -> int"""
        pass


class VectorDesCre:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorDesCre) -> None

2. __init__(self: block.operator.VectorDesCre, arg0: block.operator.VectorDesCre) -> None

Copy constructor

3. __init__(self: block.operator.VectorDesCre, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorDesCre, arg0: block.operator.VectorDesCre) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorDesCre, arg0: block.operator.VectorDesCre) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorDesCre, x: block.operator.OperatorDesCre) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorDesCre, x: block.operator.OperatorDesCre) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorDesCre, x: block.operator.OperatorDesCre) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorDesCre) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorDesCre, x: block.operator.OperatorDesCre) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorDesCre, L: block.operator.VectorDesCre) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorDesCre, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorDesCre, i: int, x: block.operator.OperatorDesCre) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorDesCre) -> block.operator.OperatorDesCre

Remove and return the last item

2. pop(self: block.operator.VectorDesCre, i: int) -> block.operator.OperatorDesCre

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorDesCre, arg0: int, arg1: block.operator.OperatorDesCre) -> None

2. __setitem__(self: block.operator.VectorDesCre, arg0: slice, arg1: block.operator.VectorDesCre) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorDesCre, s: slice) -> block.operator.VectorDesCre

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorDesCre, arg0: int) -> block.operator.OperatorDesCre"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorDesCre, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorDesCre, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorDesCre) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorDesCre) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorDesCre) -> int"""
        pass


class VectorDesCreComp:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorDesCreComp) -> None

2. __init__(self: block.operator.VectorDesCreComp, arg0: block.operator.VectorDesCreComp) -> None

Copy constructor

3. __init__(self: block.operator.VectorDesCreComp, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorDesCreComp, arg0: block.operator.VectorDesCreComp) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorDesCreComp, arg0: block.operator.VectorDesCreComp) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorDesCreComp, x: block.operator.OperatorDesCreComp) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorDesCreComp, x: block.operator.OperatorDesCreComp) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorDesCreComp, x: block.operator.OperatorDesCreComp) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorDesCreComp) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorDesCreComp, x: block.operator.OperatorDesCreComp) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorDesCreComp, L: block.operator.VectorDesCreComp) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorDesCreComp, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorDesCreComp, i: int, x: block.operator.OperatorDesCreComp) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorDesCreComp) -> block.operator.OperatorDesCreComp

Remove and return the last item

2. pop(self: block.operator.VectorDesCreComp, i: int) -> block.operator.OperatorDesCreComp

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorDesCreComp, arg0: int, arg1: block.operator.OperatorDesCreComp) -> None

2. __setitem__(self: block.operator.VectorDesCreComp, arg0: slice, arg1: block.operator.VectorDesCreComp) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorDesCreComp, s: slice) -> block.operator.VectorDesCreComp

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorDesCreComp, arg0: int) -> block.operator.OperatorDesCreComp"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorDesCreComp, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorDesCreComp, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorDesCreComp) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorDesCreComp) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorDesCreComp) -> int"""
        pass


class VectorDesDes:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorDesDes) -> None

2. __init__(self: block.operator.VectorDesDes, arg0: block.operator.VectorDesDes) -> None

Copy constructor

3. __init__(self: block.operator.VectorDesDes, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorDesDes, arg0: block.operator.VectorDesDes) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorDesDes, arg0: block.operator.VectorDesDes) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorDesDes, x: block.operator.OperatorDesDes) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorDesDes, x: block.operator.OperatorDesDes) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorDesDes, x: block.operator.OperatorDesDes) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorDesDes) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorDesDes, x: block.operator.OperatorDesDes) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorDesDes, L: block.operator.VectorDesDes) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorDesDes, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorDesDes, i: int, x: block.operator.OperatorDesDes) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorDesDes) -> block.operator.OperatorDesDes

Remove and return the last item

2. pop(self: block.operator.VectorDesDes, i: int) -> block.operator.OperatorDesDes

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorDesDes, arg0: int, arg1: block.operator.OperatorDesDes) -> None

2. __setitem__(self: block.operator.VectorDesDes, arg0: slice, arg1: block.operator.VectorDesDes) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorDesDes, s: slice) -> block.operator.VectorDesDes

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorDesDes, arg0: int) -> block.operator.OperatorDesDes"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorDesDes, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorDesDes, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorDesDes) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorDesDes) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorDesDes) -> int"""
        pass


class VectorDesDesComp:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorDesDesComp) -> None

2. __init__(self: block.operator.VectorDesDesComp, arg0: block.operator.VectorDesDesComp) -> None

Copy constructor

3. __init__(self: block.operator.VectorDesDesComp, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorDesDesComp, arg0: block.operator.VectorDesDesComp) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorDesDesComp, arg0: block.operator.VectorDesDesComp) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorDesDesComp, x: block.operator.OperatorDesDesComp) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorDesDesComp, x: block.operator.OperatorDesDesComp) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorDesDesComp, x: block.operator.OperatorDesDesComp) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorDesDesComp) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorDesDesComp, x: block.operator.OperatorDesDesComp) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorDesDesComp, L: block.operator.VectorDesDesComp) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorDesDesComp, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorDesDesComp, i: int, x: block.operator.OperatorDesDesComp) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorDesDesComp) -> block.operator.OperatorDesDesComp

Remove and return the last item

2. pop(self: block.operator.VectorDesDesComp, i: int) -> block.operator.OperatorDesDesComp

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorDesDesComp, arg0: int, arg1: block.operator.OperatorDesDesComp) -> None

2. __setitem__(self: block.operator.VectorDesDesComp, arg0: slice, arg1: block.operator.VectorDesDesComp) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorDesDesComp, s: slice) -> block.operator.VectorDesDesComp

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorDesDesComp, arg0: int) -> block.operator.OperatorDesDesComp"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorDesDesComp, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorDesDesComp, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorDesDesComp) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorDesDesComp) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorDesDesComp) -> int"""
        pass


class VectorHamiltonian:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorHamiltonian) -> None

2. __init__(self: block.operator.VectorHamiltonian, arg0: block.operator.VectorHamiltonian) -> None

Copy constructor

3. __init__(self: block.operator.VectorHamiltonian, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorHamiltonian, arg0: block.operator.VectorHamiltonian) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorHamiltonian, arg0: block.operator.VectorHamiltonian) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorHamiltonian, x: block.operator.OperatorHamiltonian) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorHamiltonian, x: block.operator.OperatorHamiltonian) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorHamiltonian, x: block.operator.OperatorHamiltonian) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorHamiltonian) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorHamiltonian, x: block.operator.OperatorHamiltonian) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorHamiltonian, L: block.operator.VectorHamiltonian) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorHamiltonian, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorHamiltonian, i: int, x: block.operator.OperatorHamiltonian) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorHamiltonian) -> block.operator.OperatorHamiltonian

Remove and return the last item

2. pop(self: block.operator.VectorHamiltonian, i: int) -> block.operator.OperatorHamiltonian

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorHamiltonian, arg0: int, arg1: block.operator.OperatorHamiltonian) -> None

2. __setitem__(self: block.operator.VectorHamiltonian, arg0: slice, arg1: block.operator.VectorHamiltonian) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorHamiltonian, s: slice) -> block.operator.VectorHamiltonian

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorHamiltonian, arg0: int) -> block.operator.OperatorHamiltonian"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorHamiltonian, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorHamiltonian, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorHamiltonian) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorHamiltonian) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorHamiltonian) -> int"""
        pass


class VectorNonZeroStackMatrix:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorNonZeroStackMatrix) -> None

2. __init__(self: block.operator.VectorNonZeroStackMatrix, arg0: block.operator.VectorNonZeroStackMatrix) -> None

Copy constructor

3. __init__(self: block.operator.VectorNonZeroStackMatrix, arg0: iterable) -> None"""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorNonZeroStackMatrix, x: Tuple[Tuple[int, int], block.operator.StackMatrix]) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorNonZeroStackMatrix, L: block.operator.VectorNonZeroStackMatrix) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorNonZeroStackMatrix, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorNonZeroStackMatrix, i: int, x: Tuple[Tuple[int, int], block.operator.StackMatrix]) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorNonZeroStackMatrix) -> Tuple[Tuple[int, int], block.operator.StackMatrix]

Remove and return the last item

2. pop(self: block.operator.VectorNonZeroStackMatrix, i: int) -> Tuple[Tuple[int, int], block.operator.StackMatrix]

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorNonZeroStackMatrix, arg0: int, arg1: Tuple[Tuple[int, int], block.operator.StackMatrix]) -> None

2. __setitem__(self: block.operator.VectorNonZeroStackMatrix, arg0: slice, arg1: block.operator.VectorNonZeroStackMatrix) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorNonZeroStackMatrix, s: slice) -> block.operator.VectorNonZeroStackMatrix

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorNonZeroStackMatrix, arg0: int) -> Tuple[Tuple[int, int], block.operator.StackMatrix]"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorNonZeroStackMatrix, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorNonZeroStackMatrix, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorNonZeroStackMatrix) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorNonZeroStackMatrix) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorNonZeroStackMatrix) -> int"""
        pass


class VectorOperatorArrayBase:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorOperatorArrayBase) -> None

2. __init__(self: block.operator.VectorOperatorArrayBase, arg0: block.operator.VectorOperatorArrayBase) -> None

Copy constructor

3. __init__(self: block.operator.VectorOperatorArrayBase, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorOperatorArrayBase, arg0: block.operator.VectorOperatorArrayBase) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorOperatorArrayBase, arg0: block.operator.VectorOperatorArrayBase) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorOperatorArrayBase, x: block.operator.OperatorArrayBase) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorOperatorArrayBase, x: block.operator.OperatorArrayBase) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorOperatorArrayBase, x: block.operator.OperatorArrayBase) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorOperatorArrayBase) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorOperatorArrayBase, x: block.operator.OperatorArrayBase) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorOperatorArrayBase, L: block.operator.VectorOperatorArrayBase) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorOperatorArrayBase, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorOperatorArrayBase, i: int, x: block.operator.OperatorArrayBase) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorOperatorArrayBase) -> block.operator.OperatorArrayBase

Remove and return the last item

2. pop(self: block.operator.VectorOperatorArrayBase, i: int) -> block.operator.OperatorArrayBase

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorOperatorArrayBase, arg0: int, arg1: block.operator.OperatorArrayBase) -> None

2. __setitem__(self: block.operator.VectorOperatorArrayBase, arg0: slice, arg1: block.operator.VectorOperatorArrayBase) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorOperatorArrayBase, s: slice) -> block.operator.VectorOperatorArrayBase

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorOperatorArrayBase, arg0: int) -> block.operator.OperatorArrayBase"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorOperatorArrayBase, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorOperatorArrayBase, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorOperatorArrayBase) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorOperatorArrayBase) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorOperatorArrayBase) -> int"""
        pass


class VectorOverlap:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorOverlap) -> None

2. __init__(self: block.operator.VectorOverlap, arg0: block.operator.VectorOverlap) -> None

Copy constructor

3. __init__(self: block.operator.VectorOverlap, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorOverlap, arg0: block.operator.VectorOverlap) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorOverlap, arg0: block.operator.VectorOverlap) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorOverlap, x: block.operator.OperatorOverlap) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorOverlap, x: block.operator.OperatorOverlap) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorOverlap, x: block.operator.OperatorOverlap) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorOverlap) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorOverlap, x: block.operator.OperatorOverlap) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorOverlap, L: block.operator.VectorOverlap) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorOverlap, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorOverlap, i: int, x: block.operator.OperatorOverlap) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorOverlap) -> block.operator.OperatorOverlap

Remove and return the last item

2. pop(self: block.operator.VectorOverlap, i: int) -> block.operator.OperatorOverlap

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorOverlap, arg0: int, arg1: block.operator.OperatorOverlap) -> None

2. __setitem__(self: block.operator.VectorOverlap, arg0: slice, arg1: block.operator.VectorOverlap) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorOverlap, s: slice) -> block.operator.VectorOverlap

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorOverlap, arg0: int) -> block.operator.OperatorOverlap"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorOverlap, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorOverlap, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorOverlap) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorOverlap) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorOverlap) -> int"""
        pass


class VectorStackSparseMatrix:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorStackSparseMatrix) -> None

2. __init__(self: block.operator.VectorStackSparseMatrix, arg0: block.operator.VectorStackSparseMatrix) -> None

Copy constructor

3. __init__(self: block.operator.VectorStackSparseMatrix, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.operator.VectorStackSparseMatrix, arg0: block.operator.VectorStackSparseMatrix) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.operator.VectorStackSparseMatrix, arg0: block.operator.VectorStackSparseMatrix) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.operator.VectorStackSparseMatrix, x: block.operator.StackSparseMatrix) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.operator.VectorStackSparseMatrix, x: block.operator.StackSparseMatrix) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.operator.VectorStackSparseMatrix, x: block.operator.StackSparseMatrix) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorStackSparseMatrix) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorStackSparseMatrix, x: block.operator.StackSparseMatrix) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorStackSparseMatrix, L: block.operator.VectorStackSparseMatrix) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorStackSparseMatrix, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorStackSparseMatrix, i: int, x: block.operator.StackSparseMatrix) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorStackSparseMatrix) -> block.operator.StackSparseMatrix

Remove and return the last item

2. pop(self: block.operator.VectorStackSparseMatrix, i: int) -> block.operator.StackSparseMatrix

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorStackSparseMatrix, arg0: int, arg1: block.operator.StackSparseMatrix) -> None

2. __setitem__(self: block.operator.VectorStackSparseMatrix, arg0: slice, arg1: block.operator.VectorStackSparseMatrix) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorStackSparseMatrix, s: slice) -> block.operator.VectorStackSparseMatrix

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorStackSparseMatrix, arg0: int) -> block.operator.StackSparseMatrix"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorStackSparseMatrix, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorStackSparseMatrix, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorStackSparseMatrix) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorStackSparseMatrix) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorStackSparseMatrix) -> int"""
        pass


class VectorWavefunction:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.operator.VectorWavefunction) -> None

2. __init__(self: block.operator.VectorWavefunction, arg0: block.operator.VectorWavefunction) -> None

Copy constructor

3. __init__(self: block.operator.VectorWavefunction, arg0: iterable) -> None"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.operator.VectorWavefunction) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.operator.VectorWavefunction, x: block.operator.Wavefunction) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.operator.VectorWavefunction, L: block.operator.VectorWavefunction) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.operator.VectorWavefunction, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.operator.VectorWavefunction, i: int, x: block.operator.Wavefunction) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.operator.VectorWavefunction) -> block.operator.Wavefunction

Remove and return the last item

2. pop(self: block.operator.VectorWavefunction, i: int) -> block.operator.Wavefunction

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.operator.VectorWavefunction, arg0: int, arg1: block.operator.Wavefunction) -> None

2. __setitem__(self: block.operator.VectorWavefunction, arg0: slice, arg1: block.operator.VectorWavefunction) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.operator.VectorWavefunction, s: slice) -> block.operator.VectorWavefunction

Retrieve list elements using a slice object

2. __getitem__(self: block.operator.VectorWavefunction, arg0: int) -> block.operator.Wavefunction"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.operator.VectorWavefunction, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.operator.VectorWavefunction, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.operator.VectorWavefunction) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.operator.VectorWavefunction) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.operator.VectorWavefunction) -> int"""
        pass


class Wavefunction(StackSparseMatrix):
    """Block-sparse matrix. 
Non-zero blocks are identified by symmetry (quantum numbers) requirements and stored as :class:`StackMatrix` objects"""

    @property
    def onedot(self):
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.operator.Wavefunction) -> None"""
        pass

    def initialize(self, *args, **kwargs):
        """initialize(self: block.operator.Wavefunction, arg0: block.symmetry.VectorSpinQuantum, arg1: block.symmetry.StateInfo, arg2: block.symmetry.StateInfo, arg3: bool) -> None"""
        pass

    def save_wavefunction_info(self, *args, **kwargs):
        """save_wavefunction_info(self: block.operator.Wavefunction, arg0: block.symmetry.StateInfo, arg1: block.VectorInt, arg2: int) -> None"""
        pass

