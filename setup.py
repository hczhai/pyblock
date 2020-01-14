#!/usr/bin/env python3

import os
import sys
import subprocess
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Label the Extension object with the directory which contains 'CMakeLists.txt'


class CMakeExt(Extension):
    def __init__(self, name, cmdir='.'):
        Extension.__init__(self, name, [])
        self.cmake_lists_dir = os.path.abspath(cmdir)

# Invoke 'cmake' to compile extension source codes.


class CMakeBuild(build_ext):
    def build_extensions(self):
        # Ensure that CMake is present and working
        try:
            out = subprocess.check_output(['cmake', '--version'])
            print(out.decode("utf-8"))
        except OSError:
            raise RuntimeError('Cannot find CMake executable!')

        print('Python3: ', sys.executable)
        print('Build Dir: ', self.build_temp)

        for ext in self.extensions:

            extdir = os.path.abspath(os.path.dirname(
                self.get_ext_fullpath(ext.name)))
            cfg = 'Release'

            cmake_args = [
                '-DCMAKE_BUILD_TYPE=%s' % cfg,
                # Ask CMake to place the resulting library in the directory
                # containing the extension
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), extdir),
                # Other intermediate static libraries are placed in a
                # temporary build directory instead
                '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), self.build_temp),
                # Hint CMake to use the same Python executable that
                # is launching the build, prevents possible mismatching if
                # multiple versions of Python are installed
                '-DPYTHON_EXECUTABLE_HINT={}'.format(sys.executable),
                # Add other project-specific CMake arguments if needed
                # ...
                '-DBUILD_LIB=ON'
            ]

            # We can handle some platform-specific settings at our discretion
            if platform.system() == 'Windows':
                plat = ('x64' if platform.architecture()
                        [0] == '64bit' else 'Win32')
                cmake_args += [
                    # These options are likely to be needed under Windows
                    '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE',
                    '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(
                        cfg.upper(), extdir)
                ]
                # Assuming that Visual Studio and MinGW are supported compilers
                if self.compiler.compiler_type == 'msvc':
                    cmake_args += [
                        '-DCMAKE_GENERATOR_PLATFORM=%s' % plat,
                    ]
                else:
                    cmake_args += [
                        '-G', 'MinGW Makefiles',
                    ]

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            # Config
            subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args,
                                  cwd=self.build_temp)

            # Build
            subprocess.check_call(['cmake', '--build', '.', '--config', cfg, '--', '--jobs=4'],
                                  cwd=self.build_temp)


setup(name='pyblock',
      version='0.1',
      description='Block (spin adapted)',
      long_description="""Based on : Block  Copyright (C) 2012  Garnet K.-L. Chan
      This program comes with ABSOLUTELY NO WARRANTY; for details see license file.
      This is free software, and you are welcome to redistribute it
      under certain conditions; see license file for details.""",
      author='Huanchen Zhai',
      author_email='hczhai@ucla.edu',
      url='-',
      ext_modules=[CMakeExt('block')],
      py_modules=['pyblock'],
      cmdclass={'build_ext': CMakeBuild}
      )
