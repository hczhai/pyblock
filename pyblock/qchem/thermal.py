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
Setting up integral for calculating thermal quantities.
"""

class FreeEnergy:
    def __init__(self, hamil):
        self.hamil = hamil
        if hamil.spin_adapted:
            self.v = hamil.v
            self.t = hamil.t
            self.e = hamil.e
            self.zv = hamil.v.__class__(hamil.v.n)
            self.zt = hamil.t.__class__(hamil.t.n)
        else:
            self.vs = hamil.vaa, hamil.vab, hamil.vba, hamil.vbb
            self.ts = hamil.ta, hamil.tb
            self.e = hamil.e
            self.zv = hamil.vaa.__class__(hamil.vaa.n)
            self.zt = hamil.ta.__class__(hamil.ta.n)
    
    def set_energy(self):
        if self.hamil.spin_adapted:
            self.hamil.v = self.v
            self.hamil.t = self.t
            self.hamil.e = self.e
        else:
            self.hamil.vaa, self.hamil.vab, self.hamil.vba, self.hamil.vbb = self.vs
            self.hamil.ta, self.hamil.tb = self.ts
            self.hamil.e = self.e
    
    def set_free_energy(self, mu):
        self.set_energy()
        if self.hamil.spin_adapted:
            self.hamil.t = self.hamil.t.copy()
            for i in range(self.hamil.n_sites):
                self.hamil.t[i, i] -= mu
        else:
            self.hamil.ta = self.hamil.ta.copy()
            self.hamil.tb = self.hamil.tb.copy()
            for t in [self.hamil.ta, self.hamil.tb]:
                for i in range(self.hamil.n_sites):
                    t[i, i] -= mu

    def set_particle_number(self):
        if self.hamil.spin_adapted:
            self.hamil.t = self.zt.copy()
            self.hamil.v = self.zv
            self.hamil.e = 0.0
            for i in range(self.hamil.n_sites):
                self.hamil.t[i, i] = 1
        else:
            self.hamil.ta = self.zt.copy()
            self.hamil.tb = self.zt.copy()
            self.hamil.vaa, self.hamil.vab, self.hamil.vba, self.hamil.vbb = [self.zv] * 4
            self.hamil.e = 0.0
            for t in [self.hamil.ta, self.hamil.tb]:
                for i in range(self.hamil.n_sites):
                    t[i, i] = 1
