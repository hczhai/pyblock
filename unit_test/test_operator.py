
from pyblock.qchem.operator import OpElement, OpNames, OpString, OpSum, op_names

import numpy as np
import pytest
import warnings
import os
from fractions import Fraction

class TestOperator:
    
    def test_operator(self):
        for x, y in op_names:
            assert repr(y) == x
        ops = []
        for x in ['H', 'I', 'C0', 'P(1, 2, 3)', 'Q(1, 1)']:
            op = OpElement.parse(x)
            ops.append(op)
            assert repr(op) == x
            assert op * (-5.0) == (-5.0) * op == 2.5 * op * (-2.0)
            assert repr(op * 2.0) == repr(2.0 * op)
            assert repr(-1.0 * op * op) == repr(op * (-1.0) * op) == repr(op * op * (-1.0))
            assert hash(op * 2.0) == hash(2.0 * op)
            assert hash(op * 2.0) != hash(2.0)
            assert op * 0 == 0 * op == 0
            assert op + 0 == op == 0 + op
            with pytest.raises(TypeError):
                op = op + type(op)
            with pytest.raises(TypeError):
                op = type(op) + op
            with pytest.raises(TypeError):
                op = op * type(op)
            with pytest.raises(TypeError):
                op = type(op) * op
        assert len(set(map(hash, ops))) == len(ops)
        a, b, c = ops[2:]
        d = a * 0.5
        e = b * 0.2
        assert repr(a * b) != repr(b * a)
        assert repr(a + b) != repr(b + a)
        assert a * b / 1.0 == a * b == abs(a * b / 2.0)
        assert d * e / 2.0 == d * e * 0.5 == 0.5 * (d * e)
        assert d * e * 0 == 0 == 0 * d * e == 0 * (d * e)
        with pytest.warns(RuntimeWarning):
            op = a * b / 0.0
        with pytest.raises(TypeError):
            op = a * b / type(a * b)
        with pytest.raises(TypeError):
            op = a * b * type(a * b)
        with pytest.raises(TypeError):
            op = type(a * b) * (a * b)
        with pytest.raises(TypeError):
            op = a * b + type(a * b)
        with pytest.raises(TypeError):
            op = type(a * b) + (a * b)
        assert c * d + 0 == 0 + c * d
        assert c * d + a * b + e * a == c * d + (a * b + e * a) == (c * d + a * b) + e * a
        assert a + b + c == (a + b) + c == a + (b + c)
        assert a + b + c + d + e == (a + b + c) + (d + e) == (a + b) + c + (d + e)
        assert a * b * c * d * e == (a * b * c) * (d * e) == (a * b) * c * (d * e)
        assert a * b * c == (a * b) * c == a * (b * c)
        assert a + b != a + b + c
        assert a * b != a * b * c
        assert a * b != b * a
        assert a != 0 and b != 0 and c != 0
        assert a * b != 0 and 0 != a * b and a * b != a
        assert a + b != 0 and 0 != a + b and a + b != b
        assert 0.5 * (a + b) == 0.5 * a + 0.5 * b
        assert repr(a + b + e) == repr(a + (b + e))
        assert (a * b) + c * d + 0 == 0 + ((a * b) + c * d) == (a * b) + c * d
        assert a + b != type(a + b) and type(a + b) != a + b
        assert a * b != type(a * b) and type(a * b) != a * b
        assert a + b != 0 and a * b != 0
        with pytest.raises(TypeError):
            op = a + b + type(a + b)
        with pytest.raises(TypeError):
            op = type(a + b) + (a + b)
        with pytest.warns(RuntimeWarning):
            op = (a + b) / 0.0
        with pytest.raises(TypeError):
            op = (a + b) / type(a + b)
        with pytest.raises(TypeError):
            op = (a + b) * type(a + b)
        with pytest.raises(TypeError):
            op = type(a + b) * (a + b)
        assert a * (c + d) == a * c + a * d
        assert (a + c + e) * d == a * d + c * d + e * d
        assert (a + c + e) * 0 == 0 * (a + c + e) == 0
        assert (a + c + e) * (-0.5) == (-0.5) * (a + c + e) == (a + c + e) / (-2.0)
        assert (a + c).strings[0].op == a
