
"""Block definition and operator operations."""

import enum

def init_big_block(*args, **kwargs):
    """init_big_block(left_block: block.block.Block, right_block: block.block.Block, big_block: block.block.Block, bra_quanta: block.symmetry.VectorSpinQuantum = VectorSpinQuantum[], ket_quanta: block.symmetry.VectorSpinQuantum = VectorSpinQuantum[]) -> None

Initialize big (super) block."""
    pass


def init_new_environment_block(*args, **kwargs):
    """init_new_environment_block(environment: block.block.Block, environment_dot: block.block.Block, new_environment: block.block.Block, system: block.block.Block, system_dot: block.block.Block, left_state: int, right_state: int, sys_add: int, env_add: int, forward: bool, direct: bool, one_dot: bool, use_slater: bool, integral_index: int, have_norm_ops: bool, have_comp_ops: bool, dot_with_sys: bool) -> None

Initialize new environment block"""
    pass


def init_new_system_block(*args, **kwargs):
    """init_new_system_block(system: block.block.Block, system_dot: block.block.Block, new_system: block.block.Block, left_state: int, right_state: int, sys_add: int, direct: bool, integral_index: int, storage: block.block.StorageTypes, have_norm_ops: bool, have_comp_ops: bool) -> None

Initialize new system block"""
    pass


def init_starting_block(*args, **kwargs):
    """init_starting_block(starting_block: block.block.Block, forward: bool, left_state: int, right_state: int, forward_starting_size: int, backward_starting_size: int, restart_size: int, restart: bool, warm_up: bool, integral_index: int, bra_quanta: block.symmetry.VectorSpinQuantum = VectorSpinQuantum[], ket_quanta: block.symmetry.VectorSpinQuantum = VectorSpinQuantum[]) -> None

Initialize starting block"""
    pass


class Block:

    @property
    def name(self):
        """A random integer."""
        pass

    @property
    def sites(self):
        """List of indices of sites contained in the block."""
        pass

    @property
    def bra_state_info(self):
        pass

    @property
    def ket_state_info(self):
        pass

    @property
    def loop_block(self):
        """Whether the block is loop block."""
        pass

    @property
    def ops(self):
        """Map from operator types to matrix representation of operators."""
        pass

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.block.Block) -> None

2. __init__(self: block.block.Block, start: int, end: int, integral_index: int, implicit_transpose: bool, is_complement: bool = False) -> None"""
        pass

    def print_operator_summary(self, *args, **kwargs):
        """print_operator_summary(self: block.block.Block) -> None

Print operator summary when :attr:`block.io.Input.output_level` at least = 2."""
        pass

    def store(self, *args, **kwargs):
        """store(self: block.block.Block, forward: bool, sites: block.VectorInt, left: int, right: int) -> None

Store a :class:`Block` into disk.

Args:
    forward : bool
        The direction of sweep.
    sites : :class:`block.VectorInt`
        List of indices of sites contained in the block. This is kind of redundant and can be obtained from :attr:`Block.sites`.
    block : :class:`Block`
        The block to store.
    left : int
        Bra state.
    right : int
        Ket state."""
        pass

    def deallocate(self, *args, **kwargs):
        """deallocate(self: block.block.Block) -> None"""
        pass

    def clear(self, *args, **kwargs):
        """clear(self: block.block.Block) -> None"""
        pass

    def transform_operators(self, *args, **kwargs):
        """transform_operators(self: block.block.Block, arg0: block.VectorMatrix) -> None"""
        pass

    def transform_operators_2(self, *args, **kwargs):
        """transform_operators_2(self: block.block.Block, left_rotate_matrix: block.VectorMatrix, right_rotate_matrix: block.VectorMatrix, clear_right_block: bool = True, clear_left_block: bool = True) -> None"""
        pass

    def move_and_free_memory(self, *args, **kwargs):
        """move_and_free_memory(self: block.block.Block, arg0: block.block.Block) -> None

If the parameter ``system`` is allocated before ``this`` object, but we need to free ``system``. Then we have to move the memory of ``this`` to ``system`` then clear ``system``."""
        pass

    def add_additional_ops(self, *args, **kwargs):
        """add_additional_ops(self: block.block.Block) -> None"""
        pass

    def remove_additional_ops(self, *args, **kwargs):
        """remove_additional_ops(self: block.block.Block) -> None"""
        pass

    def add_all_comp_ops(self, *args, **kwargs):
        """add_all_comp_ops(self: block.block.Block) -> None"""
        pass

    def multiply_overlap(self, *args, **kwargs):
        """multiply_overlap(self: block.block.Block, c: block.operator.Wavefunction, v: block.operator.Wavefunction, num_threads: int = 1) -> None"""
        pass

    def renormalize_from(self, *args, **kwargs):
        """renormalize_from(self: block.block.Block, energies: block.VectorDouble, spins: block.VectorDouble, error: float, rotate_matrix: block.VectorMatrix, kept_states: int, kept_qstates: int, tol: float, big: block.block.Block, guess_wave_type: block.block.GuessWaveTypes, noise: float, additional_noise: float, one_dot: bool, system: block.block.Block, system_dot: block.block.Block, environment: block.block.Block, dot_with_sys: bool, warm_up: bool, sweep_iter: int, current_root: int, lower_states: block.operator.VectorWavefunction) -> float"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.block.Block) -> str"""
        pass


class GuessWaveTypes(enum.Enum):
    """Types of guess wavefunction for initialize Davidson algorithm (enumerator).

Members:

  Basic

  Transform

  Transpose"""
    Basic = enum.auto()
    Transform = enum.auto()
    Transpose = enum.auto()


class MapOperators:

    def __init__(self, *args, **kwargs):
        """__init__(self: block.block.MapOperators) -> None"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.block.MapOperators) -> str

Return the canonical string representation of this map."""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.block.MapOperators) -> bool

Check whether the map is nonempty"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.block.MapOperators) -> iterator"""
        pass

    def items(self, *args, **kwargs):
        """items(self: block.block.MapOperators) -> iterator"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self: block.block.MapOperators, arg0: block.operator.OpTypes) -> block.operator.OperatorArrayBase"""
        pass

    def __contains__(self, *args, **kwargs):
        """__contains__(self: block.block.MapOperators, arg0: block.operator.OpTypes) -> bool"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self: block.block.MapOperators, arg0: block.operator.OpTypes, arg1: block.operator.OperatorArrayBase) -> None"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(self: block.block.MapOperators, arg0: block.operator.OpTypes) -> None"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.block.MapOperators) -> int"""
        pass


class StorageTypes(enum.Enum):
    """Types of storage (enumerator).

Members:

  LocalStorage

  DistributedStorage"""
    LocalStorage = enum.auto()
    DistributedStorage = enum.auto()


class VectorBlock:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.block.VectorBlock) -> None

2. __init__(self: block.block.VectorBlock, arg0: block.block.VectorBlock) -> None

Copy constructor

3. __init__(self: block.block.VectorBlock, arg0: iterable) -> None"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.block.VectorBlock) -> str

Return the canonical string representation of this list."""
        pass

    def append(self, *args, **kwargs):
        """append(self: block.block.VectorBlock, x: block.block.Block) -> None

Add an item to the end of the list"""
        pass

    def extend(self, *args, **kwargs):
        """extend(*args, **kwargs)
Overloaded function.

1. extend(self: block.block.VectorBlock, L: block.block.VectorBlock) -> None

Extend the list by appending all the items in the given list

2. extend(self: block.block.VectorBlock, L: iterable) -> None

Extend the list by appending all the items in the given list"""
        pass

    def insert(self, *args, **kwargs):
        """insert(self: block.block.VectorBlock, i: int, x: block.block.Block) -> None

Insert an item at a given position."""
        pass

    def pop(self, *args, **kwargs):
        """pop(*args, **kwargs)
Overloaded function.

1. pop(self: block.block.VectorBlock) -> block.block.Block

Remove and return the last item

2. pop(self: block.block.VectorBlock, i: int) -> block.block.Block

Remove and return the item at index ``i``"""
        pass

    def __setitem__(self, *args, **kwargs):
        """__setitem__(*args, **kwargs)
Overloaded function.

1. __setitem__(self: block.block.VectorBlock, arg0: int, arg1: block.block.Block) -> None

2. __setitem__(self: block.block.VectorBlock, arg0: slice, arg1: block.block.VectorBlock) -> None

Assign list elements using a slice object"""
        pass

    def __getitem__(self, *args, **kwargs):
        """__getitem__(*args, **kwargs)
Overloaded function.

1. __getitem__(self: block.block.VectorBlock, s: slice) -> block.block.VectorBlock

Retrieve list elements using a slice object

2. __getitem__(self: block.block.VectorBlock, arg0: int) -> block.block.Block"""
        pass

    def __delitem__(self, *args, **kwargs):
        """__delitem__(*args, **kwargs)
Overloaded function.

1. __delitem__(self: block.block.VectorBlock, arg0: int) -> None

Delete the list elements at index ``i``

2. __delitem__(self: block.block.VectorBlock, arg0: slice) -> None

Delete list elements using a slice object"""
        pass

    def __iter__(self, *args, **kwargs):
        """__iter__(self: block.block.VectorBlock) -> iterator"""
        pass

    def __bool__(self, *args, **kwargs):
        """__bool__(self: block.block.VectorBlock) -> bool

Check whether the list is nonempty"""
        pass

    def __len__(self, *args, **kwargs):
        """__len__(self: block.block.VectorBlock) -> int"""
        pass

