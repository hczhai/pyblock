
"""Classes for handling symmetries and quantum numbers."""


def get_commute_parity(*args, **kwargs):
    """get_commute_parity(a: block.symmetry.SpinQuantum, b: block.symmetry.SpinQuantum, c: block.symmetry.SpinQuantum) -> float"""
    pass


def state_tensor_product(*args, **kwargs):
    """state_tensor_product(arg0: block.symmetry.StateInfo, arg1: block.symmetry.StateInfo) -> block.symmetry.StateInfo"""
    pass


def state_tensor_product_target(*args, **kwargs):
    """state_tensor_product_target(arg0: block.symmetry.StateInfo, arg1: block.symmetry.StateInfo) -> block.symmetry.StateInfo"""
    pass


def wigner_9j(*args, **kwargs):
    """wigner_9j(arg0: int, arg1: int, arg2: int, arg3: int, arg4: int, arg5: int, arg6: int, arg7: int, arg8: int) -> float"""
    pass


class IrrepSpace:
    """A wrapper class for molecular point group symmetry irrep."""

    @property
    def irrep(self):
        pass

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.symmetry.IrrepSpace) -> None

2. __init__(self: block.symmetry.IrrepSpace, arg0: int) -> None"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.symmetry.IrrepSpace) -> str"""
        pass


class SpinQuantum:
    """A collection of quantum numbers associated with a specific state (irreducible representation). One such collection defines a specific sector in the state space."""

    @property
    def s(self):
        """Irreducible representation for spin symmetry (:math:`S` or :math:`S_z`)."""
        pass

    @property
    def n(self):
        """Particle number."""
        pass

    @property
    def symm(self):
        """Irreducible representation for molecular point group symmetry."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.symmetry.SpinQuantum) -> None

2. __init__(self: block.symmetry.SpinQuantum, arg0: int, arg1: block.symmetry.SpinSpace, arg2: block.symmetry.IrrepSpace) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.symmetry.SpinQuantum, arg0: block.symmetry.SpinQuantum) -> bool"""
        pass

    def __lt__(self, *args, **kwargs):
        """__lt__(self: block.symmetry.SpinQuantum, arg0: block.symmetry.SpinQuantum) -> bool"""
        pass

    def __add__(self, *args, **kwargs):
        """__add__(self: block.symmetry.SpinQuantum, arg0: block.symmetry.SpinQuantum) -> std::vector<SpinAdapted::SpinQuantum, std::allocator<SpinAdapted::SpinQuantum> >"""
        pass

    def __sub__(self, *args, **kwargs):
        """__sub__(self: block.symmetry.SpinQuantum, arg0: block.symmetry.SpinQuantum) -> std::vector<SpinAdapted::SpinQuantum, std::allocator<SpinAdapted::SpinQuantum> >"""
        pass

    def __neg__(self, *args, **kwargs):
        """__neg__(self: block.symmetry.SpinQuantum) -> block.symmetry.SpinQuantum"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.symmetry.SpinQuantum) -> str"""
        pass


class SpinSpace:
    """A wrapper class for the spin irrep.

In :math:`S_z` symmetry, the irrep is :math:`2S_z`. In SU(2) symmetry, the irrep is :math:`2S`. The behaviour is toggled checking :attr:`block.io.Global.dmrginp.spin_adapted`."""

    @property
    def irrep(self):
        pass

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.symmetry.SpinSpace) -> None

2. __init__(self: block.symmetry.SpinSpace, arg0: int) -> None"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.symmetry.SpinSpace) -> str"""
        pass


class StateInfo:
    """A collection of symmetry sectors. Each sector can contain several internal states (which can no longer be differentiated by symmetries), the number of which is also stored."""

    @property
    def quanta(self):
        """Quantum numbers for a set of sites."""
        pass

    @property
    def n_states(self):
        """Number of states per (combined) quantum number."""
        pass

    @property
    def left_unmap_quanta(self):
        """Index in left StateInfo, for each combined state."""
        pass

    @property
    def right_unmap_quanta(self):
        """Index in right StateInfo, for each combined state."""
        pass

    @property
    def old_to_new_state(self):
        """old_to_new_state[i] = [k1, k2, k3, ...] where i is the index in the collected StateInfo and k's are indices in the uncollected StateInfo."""
        pass

    @property
    def left_state_info(self):
        pass

    @property
    def right_state_info(self):
        pass

    @property
    def uncollected_state_info(self):
        pass

    @property
    def n_total_states(self):
        pass

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.symmetry.StateInfo) -> None

2. __init__(self: block.symmetry.StateInfo, arg0: block.symmetry.VectorSpinQuantum, arg1: block.VectorInt) -> None"""
        pass

    def set_left_state_info(self, *args, **kwargs):
        """set_left_state_info(self: block.symmetry.StateInfo, arg0: block.symmetry.StateInfo) -> None"""
        pass

    def set_right_state_info(self, *args, **kwargs):
        """set_right_state_info(self: block.symmetry.StateInfo, arg0: block.symmetry.StateInfo) -> None"""
        pass

    def set_uncollected_state_info(self, *args, **kwargs):
        """set_uncollected_state_info(self: block.symmetry.StateInfo, arg0: block.symmetry.StateInfo) -> None"""
        pass

    def collect_quanta(self, *args, **kwargs):
        """collect_quanta(self: block.symmetry.StateInfo) -> None"""
        pass

    def copy(self, *args, **kwargs):
        """copy(self: block.symmetry.StateInfo) -> block.symmetry.StateInfo"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.symmetry.StateInfo) -> str"""
        pass


class VectorSpinQuantum:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.symmetry.VectorSpinQuantum) -> None

2. __init__(self: block.symmetry.VectorSpinQuantum, arg0: block.symmetry.VectorSpinQuantum) -> None

Copy constructor

3. __init__(self: block.symmetry.VectorSpinQuantum, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.symmetry.VectorSpinQuantum, arg0: block.symmetry.VectorSpinQuantum) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.symmetry.VectorSpinQuantum, arg0: block.symmetry.VectorSpinQuantum) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.symmetry.VectorSpinQuantum, x: block.symmetry.SpinQuantum) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.symmetry.VectorSpinQuantum, x: block.symmetry.SpinQuantum) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.symmetry.VectorSpinQuantum, x: block.symmetry.SpinQuantum) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.symmetry.VectorSpinQuantum) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.symmetry.VectorSpinQuantum, x: block.symmetry.SpinQuantum) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.symmetry.VectorSpinQuantum, L: block.symmetry.VectorSpinQuantum) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.symmetry.VectorSpinQuantum, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.symmetry.VectorSpinQuantum, i: int, x: block.symmetry.SpinQuantum) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.symmetry.VectorSpinQuantum) -> block.symmetry.SpinQuantum

Remove and return the last item

2. pop(self: block.symmetry.VectorSpinQuantum, i: int) -> block.symmetry.SpinQuantum

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.symmetry.VectorSpinQuantum, arg0: int, arg1: block.symmetry.SpinQuantum) -> None

2. __setitem__(self: block.symmetry.VectorSpinQuantum, arg0: slice, arg1: block.symmetry.VectorSpinQuantum) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.symmetry.VectorSpinQuantum, s: slice) -> block.symmetry.VectorSpinQuantum

Retrieve list elements using a slice object

2. __getitem__(self: block.symmetry.VectorSpinQuantum, arg0: int) -> block.symmetry.SpinQuantum"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.symmetry.VectorSpinQuantum, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.symmetry.VectorSpinQuantum, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.symmetry.VectorSpinQuantum) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.symmetry.VectorSpinQuantum) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.symmetry.VectorSpinQuantum) -> int"""
        pass


class VectorStateInfo:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.symmetry.VectorStateInfo) -> None

2. __init__(self: block.symmetry.VectorStateInfo, arg0: block.symmetry.VectorStateInfo) -> None

Copy constructor

3. __init__(self: block.symmetry.VectorStateInfo, arg0: iterable) -> None"""
        pass

    def __eq__(self, *args, **kwargs):
        """__eq__(self: block.symmetry.VectorStateInfo, arg0: block.symmetry.VectorStateInfo) -> bool"""
        pass

    def __ne__(self, *args, **kwargs):
        """__ne__(self: block.symmetry.VectorStateInfo, arg0: block.symmetry.VectorStateInfo) -> bool"""
        pass

    def count(self, *args, **kwargs):
        """count(self: block.symmetry.VectorStateInfo, x: block.symmetry.StateInfo) -> int

Return the number of times ``x`` appears in the list"""
        pass

    def remove(self, *args, **kwargs):
        """remove(self: block.symmetry.VectorStateInfo, x: block.symmetry.StateInfo) -> None

Remove the first item from the list whose value is x. It is an error if there is no such item."""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.symmetry.VectorStateInfo, x: block.symmetry.StateInfo) -> bool

Return true the container contains ``x``"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.symmetry.VectorStateInfo) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.symmetry.VectorStateInfo, x: block.symmetry.StateInfo) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.symmetry.VectorStateInfo, L: block.symmetry.VectorStateInfo) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.symmetry.VectorStateInfo, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.symmetry.VectorStateInfo, i: int, x: block.symmetry.StateInfo) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.symmetry.VectorStateInfo) -> block.symmetry.StateInfo

Remove and return the last item

2. pop(self: block.symmetry.VectorStateInfo, i: int) -> block.symmetry.StateInfo

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.symmetry.VectorStateInfo, arg0: int, arg1: block.symmetry.StateInfo) -> None

2. __setitem__(self: block.symmetry.VectorStateInfo, arg0: slice, arg1: block.symmetry.VectorStateInfo) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.symmetry.VectorStateInfo, s: slice) -> block.symmetry.VectorStateInfo

Retrieve list elements using a slice object

2. __getitem__(self: block.symmetry.VectorStateInfo, arg0: int) -> block.symmetry.StateInfo"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.symmetry.VectorStateInfo, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.symmetry.VectorStateInfo, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.symmetry.VectorStateInfo) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.symmetry.VectorStateInfo) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.symmetry.VectorStateInfo) -> int"""
        pass

