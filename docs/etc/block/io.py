
"""Contains Input/Output related interfaces."""

import enum

def get_current_stack_memory(*args, **kwargs):
    """get_current_stack_memory() -> int"""
    pass


def init_stack_memory(*args, **kwargs):
    """init_stack_memory() -> None"""
    pass


def read_input(*args, **kwargs):
    """read_input(arg0: str) -> None"""
    pass


def release_stack_memory(*args, **kwargs):
    """release_stack_memory() -> None"""
    pass


def set_current_stack_memory(*args, **kwargs):
    """set_current_stack_memory(arg0: int) -> None"""
    pass


class AlgorithmTypes(enum.Enum):
    """Types of algorithm: one-dot or two-dot or other types.

Members:

  OneDot

  TwoDot

  TwoDotToOneDot

  PartialSweep"""
    OneDot = enum.auto()
    TwoDot = enum.auto()
    TwoDotToOneDot = enum.auto()
    PartialSweep = enum.auto()


class CumulTimer:

    def __init__(self, *args, **kwargs):
        """__init__(self: block.io.CumulTimer) -> None"""
        pass

    def start(self, *args, **kwargs):
        """start(self: block.io.CumulTimer) -> None"""
        pass

    def reset(self, *args, **kwargs):
        """reset(self: block.io.CumulTimer) -> None"""
        pass

    def stop(self, *args, **kwargs):
        """stop(self: block.io.CumulTimer) -> None"""
        pass

    def __repr__(self, *args, **kwargs):
        """__repr__(self: block.io.CumulTimer) -> str"""
        pass


class Global:
    """Wrapper for global variables."""
    dmrginp = None
    non_abelian_sym = None
    point_group = None


class Input:

    @property
    def output_level(self):
        pass

    @property
    def sweep_tol(self):
        pass

    @property
    def spin_orbs_symmetry(self):
        """Spatial symmetry (irrep number) of each spin-orbital."""
        pass

    @property
    def molecule_quantum(self):
        """Symmetry of target state."""
        pass

    @property
    def algorithm_type(self):
        """Algorithm type: one-dot or two-dot or other types."""
        pass

    @property
    def twodot_to_onedot_iter(self):
        """Indicating at which sweep iteration the switching from two-dot to one-dot will happen."""
        pass

    @property
    def n_max_iters(self):
        """The maximal number of sweep iterations (outer loop)."""
        pass

    @property
    def slater_size(self):
        """Number of spin-orbitals"""
        pass

    @property
    def n_electrons(self):
        """Number of electrons"""
        pass

    @property
    def timer_guessgen(self):
        """Timer for generating or loading dot blocks and environment block."""
        pass

    @property
    def timer_multiplier(self):
        """Timer for blocking."""
        pass

    @property
    def timer_operrot(self):
        """Timer for operator rotation."""
        pass

    @property
    def is_spin_adapted(self):
        """Indicates whether SU(2) symmetry is utilized. If SU(2) is not used, The Abelian subgroup of SU(2) (Sz symmetry) is used."""
        pass

    @property
    def hf_occupancy(self):
        pass

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.io.Input) -> None

2. __init__(self: block.io.Input, arg0: str) -> None

Initialize an Input object from the input file name."""
        pass

    def effective_molecule_quantum_vec(self, *args, **kwargs):
        """effective_molecule_quantum_vec(self: block.io.Input) -> block.symmetry.VectorSpinQuantum

Often this simply returns a vector containing one ``molecule_quantum``. For non-interacting orbitals or Bogoliubov algorithm, this may be more than that."""
        pass

    def n_roots(self, *args, **kwargs):
        """n_roots(self: block.io.Input, sweep_iter: int) -> int

Get number of states to solve for given sweep iteration."""
        pass


class Timer:

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.io.Timer) -> None

2. __init__(self: block.io.Timer, arg0: bool) -> None

With a `bool` parameter indicating whether the :class:`Timer` should start immediately."""
        pass

    def start(self, *args, **kwargs):
        """start(self: block.io.Timer) -> None"""
        pass

    def elapsed_walltime(self, *args, **kwargs):
        """elapsed_walltime(self: block.io.Timer) -> int"""
        pass

    def elapsed_cputime(self, *args, **kwargs):
        """elapsed_cputime(self: block.io.Timer) -> float"""
        pass

