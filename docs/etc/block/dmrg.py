
"""DMRG calculations."""


def MPS_init(*args, **kwargs):
    """MPS_init(arg0: bool) -> None

Initialize the single site blocks :attr:`MPS.site_blocks`. """
    pass


def block_and_decimate(*args, **kwargs):
    """block_and_decimate(sweep_params: block.dmrg.SweepParams, system: block.block.Block, new_system: block.block.Block, use_slater: bool, dot_with_sys: bool) -> None

Block and decimate to generate the new system block."""
    pass


def calldmrg(*args, **kwargs):
    """calldmrg(input_file_name: str) -> None

Global driver."""
    pass


def dmrg(*args, **kwargs):
    """dmrg(sweep_tol: float) -> None

Perform DMRG calculation."""
    pass


def do_one(*args, **kwargs):
    """do_one(sweep_params: block.dmrg.SweepParams, warm_up: bool, forward: bool, restart: bool, restart_size: int) -> float

Perform one sweep procedure."""
    pass


def get_dot_with_sys(*args, **kwargs):
    """get_dot_with_sys(system: block.block.Block, one_dot: bool, forward: bool) -> bool

Return the `dot_with_sys` variable, determing whether the complementary operators should be defined based on system block indicies."""
    pass


def guess_wavefunction(*args, **kwargs):
    """guess_wavefunction(solution: block.operator.Wavefunction, e: block.DiagonalMatrix, big: block.block.Block, guess_wave_type: block.block.GuessWaveTypes, one_dot: bool, state: int, transpose_guess_wave: bool, additional_noise: float = 0.0) -> None"""
    pass


def make_system_environment_big_overlap_blocks(*args, **kwargs):
    """make_system_environment_big_overlap_blocks(system_sites: block.VectorInt, system_dot: block.block.Block, environment_dot: block.block.Block, system: block.block.Block, new_system: block.block.Block, environment: block.block.Block, new_environment: block.block.Block, big: block.block.Block, sweep_params: block.dmrg.SweepParams, dot_with_sys: bool, use_slater: bool, integral_index: int, bra_state: int, ket_state: int) -> None"""
    pass


class MPS:
    site_blocks = None
    n_sweep_iters = None

    def __init__(self, *args, **kwargs):
        """__init__(*args, **kwargs)
Overloaded function.

1. __init__(self: block.dmrg.MPS) -> None

2. __init__(self: block.dmrg.MPS, arg0: block.VectorBool) -> None"""
        pass

    def get_site_tensors(self, *args, **kwargs):
        """get_site_tensors(self: block.dmrg.MPS, arg0: int) -> block.VectorMatrix"""
        pass

    def get_w(self, *args, **kwargs):
        """get_w(self: block.dmrg.MPS) -> block.operator.Wavefunction"""
        pass

    def write_to_disk(self, *args, **kwargs):
        """write_to_disk(self: block.dmrg.MPS, state_index: int, write_state_average: bool = False) -> None"""
        pass


class SweepParams:

    @property
    def current_root(self):
        pass

    @property
    def sweep_iter(self):
        """Counter for controlling the sweep iteration (outer loop)."""
        pass

    @property
    def block_iter(self):
        """Counter for controlling the blocking iteration (inner loop)."""
        pass

    @property
    def n_block_iters(self):
        """The number of blocking iterations (inner loops) needed in one sweep."""
        pass

    @property
    def n_keep_states(self):
        """The bond dimension for states in current sweep."""
        pass

    @property
    def n_keep_qstates(self):
        """(May not be useful.)"""
        pass

    @property
    def largest_dw(self):
        """Largest discarded weight (or largest error)."""
        pass

    @property
    def lowest_energy(self):
        pass

    @property
    def lowest_energy_spin(self):
        pass

    @property
    def lowest_error(self):
        pass

    @property
    def davidson_tol(self):
        pass

    @property
    def forward_starting_size(self):
        """Initial size of system block if in forward direction."""
        pass

    @property
    def backward_starting_size(self):
        """Initial size of system block if in backward direction."""
        pass

    @property
    def sys_add(self):
        """The dot block size near system block."""
        pass

    @property
    def env_add(self):
        """The dot block size near environment block."""
        pass

    @property
    def one_dot(self):
        """Whether it is the one-dot scheme."""
        pass

    @property
    def guess_type(self):
        pass

    @property
    def noise(self):
        pass

    @property
    def additional_noise(self):
        pass

    def __init__(self, *args, **kwargs):
        """__init__(self: block.dmrg.SweepParams) -> None"""
        pass

    def set_sweep_parameters(self, *args, **kwargs):
        """set_sweep_parameters(self: block.dmrg.SweepParams) -> None"""
        pass

    def save_state(self, *args, **kwargs):
        """save_state(self: block.dmrg.SweepParams, forward: bool, size: int) -> None

Save the sweep direction and number of sites in system block into the disk file 'statefile.*.tmp'."""
        pass

