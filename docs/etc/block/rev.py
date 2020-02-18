
"""Revised Block functions."""


def product(*args, **kwargs):
    """product(a: block.operator.StackSparseMatrix, b: block.operator.StackSparseMatrix, c: block.operator.StackSparseMatrix, state_info: block.symmetry.StateInfo, scale: float = 1.0) -> None"""
    pass


def tensor_dot_product(*args, **kwargs):
    """tensor_dot_product(a: block.operator.StackSparseMatrix, b: block.operator.StackSparseMatrix) -> float"""
    pass


def tensor_precondition(*args, **kwargs):
    """tensor_precondition(a: block.operator.StackSparseMatrix, e: float, diag: block.DiagonalMatrix) -> None"""
    pass


def tensor_product(*args, **kwargs):
    """tensor_product(a: block.operator.StackSparseMatrix, b: block.operator.StackSparseMatrix, c: block.operator.StackSparseMatrix, state_info: block.symmetry.VectorStateInfo, scale: float = 1.0) -> None"""
    pass


def tensor_product_diagonal(*args, **kwargs):
    """tensor_product_diagonal(a: block.operator.StackSparseMatrix, b: block.operator.StackSparseMatrix, c: block.DiagonalMatrix, state_info: block.symmetry.VectorStateInfo, scale: float = 1.0) -> None"""
    pass


def tensor_product_multiply(*args, **kwargs):
    """tensor_product_multiply(a: block.operator.StackSparseMatrix, b: block.operator.StackSparseMatrix, c: block.operator.Wavefunction, v: block.operator.Wavefunction, state_info: block.symmetry.StateInfo, op_q: block.symmetry.SpinQuantum, scale: float) -> None"""
    pass


def tensor_rotate(*args, **kwargs):
    """tensor_rotate(a: block.operator.StackSparseMatrix, c: block.operator.StackSparseMatrix, state_info: block.symmetry.VectorStateInfo, rotate_matrix: block.VectorMatrix) -> None"""
    pass


def tensor_scale(*args, **kwargs):
    """tensor_scale(scale: float, a: block.operator.StackSparseMatrix) -> None"""
    pass


def tensor_scale_add(*args, **kwargs):
    """tensor_scale_add(scale: float, a: block.operator.StackSparseMatrix, c: block.operator.StackSparseMatrix, state_info: block.symmetry.StateInfo) -> None"""
    pass


def tensor_scale_add_no_trans(*args, **kwargs):
    """tensor_scale_add_no_trans(scale: float, a: block.operator.StackSparseMatrix, c: block.operator.StackSparseMatrix) -> None"""
    pass


def tensor_trace(*args, **kwargs):
    """tensor_trace(a: block.operator.StackSparseMatrix, c: block.operator.StackSparseMatrix, state_info: block.symmetry.VectorStateInfo, trace_right: bool, scale: float = 1.0) -> None"""
    pass


def tensor_trace_diagonal(*args, **kwargs):
    """tensor_trace_diagonal(a: block.operator.StackSparseMatrix, c: block.DiagonalMatrix, state_info: block.symmetry.VectorStateInfo, trace_right: bool, scale: float = 1.0) -> None"""
    pass


def tensor_trace_multiply(*args, **kwargs):
    """tensor_trace_multiply(a: block.operator.StackSparseMatrix, c: block.operator.Wavefunction, v: block.operator.Wavefunction, state_info: block.symmetry.StateInfo, trace_right: bool, scale: float) -> None"""
    pass

