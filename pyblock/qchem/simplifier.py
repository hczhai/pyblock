#
#    pyblock: Spin-adapted quantum chemistry DMRG in MPO language (based on Block C++ code)
#    Copyright (C) 2019-2020 Huanchen Zhai
#
#    Block 1.5.3: density matrix renormalization group (DMRG) algorithm for quantum chemistry
#    Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012
#    Copyright (C) 2012 Garnet K.-L. Chan
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Rules for simplifying symbolic operator expressions.
"""

from .operator import OpNames, OpElement, OpString, OpSum
import contextlib
import numpy as np


class OpCollection:
    def __init__(self, uniq, linked=None):
        self.uniq = uniq
        self.uniq_list = sorted(self.uniq.items(), key=lambda x: x[0])
        self.linked = linked if linked is not None else []
    
    @contextlib.contextmanager
    def __call__(self):
        new_ops = {}
        yield self.uniq_list, new_ops
        for op, _, link in self.linked:
            if op not in self.uniq:
                new_ops[op] = link(new_ops)


class NoSimplifier:
    """No simplification is performed."""
    def __init__(self):
        pass
    
    def simplify(self, zipped):
        return OpCollection(dict(zipped))


class OpShell:
    def __init__(self, data):
        self.data = data


class OpLink:
    def __init__(self, op, trans, scale):
        self.op = op
        self.trans = trans
        self.scale = scale
    
    def __call__(self, op_dict):
        if op_dict[self.op] == 0:
            return 0
        else:
            if not self.trans and self.scale == 1:
                return op_dict[self.op]
            else:
                mat = op_dict[self.op].__class__()
                mat.shallow_copy(op_dict[self.op])
                if self.trans:
                    mat.conjugacy = 'n' if mat.conjugacy == 't' else 't'
                mat.symm_scale *= self.scale
                return mat

class Rule:
    def __init__(self, f=lambda op: None):
        self.f = f
    
    def __call__(self, op):
        return self.f(op)
    
    def __or__(self, other):
        def f(op):
            x = self(op)
            return x if x is not None else other(op)
        return Rule(f)

class RuleD(Rule):
    def __call__(self, op):
        if op.name == OpNames.D:
            return OpLink(OpElement(OpNames.C, op.site_index), True, 1)
        else:
            return None

class RuleR(Rule):
    def __call__(self, op):
        if op.name == OpNames.RD:
            return OpLink(OpElement(OpNames.R, op.site_index), True, -1)
        else:
            return None

class RuleA(Rule):
    def __call__(self, op):
        if op.name == OpNames.A:
            i, j, s = op.site_index
            if i >= j:
                return None
            else:
                return OpLink(OpElement(OpNames.A, (j, i, s)), False, -1 if s == 1 else 1)
        elif op.name == OpNames.AD:
            i, j, s = op.site_index
            if i >= j:
                return OpLink(OpElement(OpNames.A, (i, j, s)), True, -1 if s == 0 else 1)
            else:
                return OpLink(OpElement(OpNames.A, (j, i, s)), True, -1)
        else:
            return None


class RuleP(Rule):
    def __call__(self, op):
        if op.name == OpNames.P:
            i, j, s = op.site_index
            if i >= j:
                return None
            else:
                return OpLink(OpElement(OpNames.P, (j, i, s)), False, -1 if s == 1 else 1)
        elif op.name == OpNames.PD:
            i, j, s = op.site_index
            if i >= j:
                return OpLink(OpElement(OpNames.P, (i, j, s)), True, -1 if s == 0 else 1)
            else:
                return OpLink(OpElement(OpNames.P, (j, i, s)), True, -1)
        else:
            return None

class RuleB(Rule):
    def __call__(self, op):
        if op.name == OpNames.B:
            i, j, s = op.site_index
            if i >= j:
                return None
            else:
                return OpLink(OpElement(OpNames.B, (j, i, s)), True, -1 if s == 1 else 1)
        else:
            return None


class RuleQ(Rule):
    def __call__(self, op):
        if op.name == OpNames.Q:
            i, j, s = op.site_index
            if i >= j:
                return None
            else:
                return OpLink(OpElement(OpNames.Q, (j, i, s)), True, -1 if s == 1 else 1)
        else:
            return None


class AllRules(Rule):
    def __init__(self):
        self.f = (RuleA() | RuleP() |  RuleB() | RuleQ() | RuleR() | RuleD()).__call__


class NoTransposeRules(Rule):
    def __init__(self, rule=AllRules()):
        self.f = rule.__call__
    
    def __call__(self, op):
        link = super().__call__(op)
        if link is None or link.trans:
            return None
        else:
            return link


class Simplifier:
    """Simplify complementary operators using symmetry properties."""
    def __init__(self, rule):
        self.rule = rule
        self.op_map = {}
    
    def simplify(self, zipped):
        uniq = {}
        linked = []
        for op, expr in zipped:
            if op not in self.op_map:
                self.op_map[op] = self.rule(op)
            if self.op_map[op] is None:
                uniq[op] = expr
            else:
                linked.append((op, expr, self.op_map[op]))
        for op, expr, link in linked:
            if link.op not in uniq:
                uniq[op] = expr
        return OpCollection(uniq, linked)
