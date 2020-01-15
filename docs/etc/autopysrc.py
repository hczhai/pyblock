
"""
Create skeleton python source code for documentation purpose.
Can be used with python module dynamically generated from pybind11.

usage:
    PYTHONPATH=/path/to/module python3 autopysrc.py <module name> <output zip filename>

for example:
    PYTHONPATH=/work/code/build-cmake python3 autopysrc.py block block.zip
"""

import inspect
import zipfile
import sys
import re

def check_module(py_module, obj, obj_name):
    return obj.__class__.__name__ == 'module' and \
        obj_name not in sys.builtin_module_names and \
        (not inspect.isbuiltin(obj) and (
        not hasattr(obj, '__file__') or \
        (py_module.__file__.replace('__init__.py', obj_name + '.py') == obj.__file__ or
            py_module.__file__.replace('__init__.py', obj_name + '/__init__.py') == obj.__file__)))


def check_function(py_module, obj, obj_name):
    return obj.__class__.__name__ == 'function' and \
        (not inspect.isbuiltin(obj) and
        (not hasattr(obj, '__file__') or inspect.getfile(obj) == py_module.__file__))

def check_class(py_module, obj, obj_name):
    return inspect.isclass(obj) and \
        (not inspect.isbuiltin(obj) and
        (not hasattr(obj, '__file__') or inspect.getfile(obj) == py_module.__file__))
    
def check_builtin_function(py_module, obj, obj_name):
    return obj.__class__.__name__ == 'builtin_function_or_method'


def parse_property(name, obj):
    r = {'name': name }
    r['doc'] = inspect.getdoc(obj)
    r['signature'] = '(self)'
    return r

def parse_function(obj, instance=False):
    r = {'name': obj.__name__}
    doc = inspect.getdoc(obj)
    if instance:
        r['signature'] = '(self, *args, **kwargs)'
    else:
        r['signature'] = '(*args, **kwargs)'
    try:
        r['signature'] = str(inspect.signature(obj))
    except:
        pass
    r['doc'] = doc
    return r

def parse_class(obj):
    r = {'name': obj.__name__}
    r['imports'] = []
    r['bases'] = []
    r['instancemethod'] = []
    r['static_property'] = []
    r['property'] = []
    r['entries'] = []
    r['doc'] = inspect.getdoc(obj)
    obj_module = obj.__module__
    for m in obj.__bases__:
        r['bases'].append(m.__qualname__)
        if m.__module__ != obj_module:
            r['bases'][-1] = m.__module__ + '.' + r['bases'][-1]
            r['imports'].append(m.__module__)
    if '__entries' in obj.__dict__:
        r['bases'].append('enum.Enum')
        r['imports'].append('enum')
        for k, v in obj.__entries.items():
            r['entries'].append(k)
    else:
        for k, v in obj.__dict__.items():
            if v.__class__.__name__ == 'instancemethod':
                r['instancemethod'].append(parse_function(v, True))
            elif v.__class__.__name__ == 'pybind11_static_property':
                r['static_property'].append(k)
            elif v.__class__.__name__ == 'property':
                r['property'].append(parse_property(k, v))
    r['imports'] = list(set(r['imports']))
    return r


def parse_module(module_name, py_module, recursive=True):
    r = {'doc': inspect.getdoc(py_module),
         'name': module_name,
         'sub_modules': [],
         'functions': [],
         'classes': []
         }
    r_sub = r['sub_modules']
    r_func = r['functions']
    r_class = r['classes']
    if recursive:
        for (name, obj) in inspect.getmembers(py_module):
            if check_module(py_module, obj, name):
                r_sub.append(parse_module(name, obj, True))
            elif check_function(py_module, obj, name):
                r_func.append(parse_function(obj))
            elif check_builtin_function(py_module, obj, name):
                r_func.append(parse_function(obj))
            elif check_class(py_module, obj, name):
                r_class.append(parse_class(obj))
    return r


def write_python(r, out_dir='.', src=None, no_buildin=True):
    if src is None:
        src = []
    doc = []
    doc.append('' if r['doc'] is None else '\n"""%s"""\n\n' % r['doc'])
    ips = set()
    for c in r['classes']:
        if c is not None and 'imports' in c:
            for ip in c['imports']:
                ips.add(ip)
    for ip in ips:
        if no_buildin and ip == 'pybind11_builtins':
            continue
        doc.append('import %s\n' % ip)
    for func in r['functions']:
        if func is None:
            continue
        doc.append('\ndef %s%s:\n' % (func['name'], func['signature']))
        if func['doc'] is not None:
            doc.append('    """%s"""\n' % func['doc'])
        doc.append('    pass\n\n')
    r['classes'] = [n for n in r['classes'] if n is not None]
    names = [n['name'] for n in r['classes']]
    solved = set()
    while len(solved) != len(r['classes']):
        for ic in range(len(r['classes'])):
            if r['classes'][ic]['name'] in solved:
                continue
            dep = False
            for b in r['classes'][ic]['bases']:
                if b not in solved and b in names:
                    dep = True
                    break
            if not dep:
                break
        c = r['classes'][ic]
        solved.add(c['name'])
        print(c['name'])
        if no_buildin:
            cb = [cc for cc in c['bases'] if not cc.startswith('pybind11_builtins.')]
        else:
            cb = c['bases']
        if cb == []:
            doc.append('\nclass %s:\n' % c['name'])
        else:
            doc.append('\nclass %s(%s):\n' % (c['name'], ", ".join(cb)))
        if c['doc'] is not None:
            doc.append('    """%s"""\n' % c['doc'])
        for func in c['property']:
            if func is None:
                continue
            doc.append('\n    @property\n    def %s%s:\n' % (func['name'], func['signature']))
            if func['doc'] is not None and len(func['doc']) != 0:
                doc.append('        """%s"""\n' % func['doc'])
            doc.append('        pass\n')
        for ent in c['entries']:
            doc.append('    %s = enum.auto()\n' % ent)
        for ent in c['static_property']:
            if no_buildin:
                doc.append('    %s = None\n' % ent)
            else:
                doc.append('    %s = pybind11_builtins.pybind11_static_property\n' % ent)
        for func in c['instancemethod']:
            if func is None:
                continue
            doc.append('\n    def %s%s:\n' % (func['name'], func['signature']))
            if func['doc'] is not None:
                doc.append('        """%s"""\n' % func['doc'])
            doc.append('        pass\n')
        if len(c['instancemethod']) == 0 and len(c['entries']) == 0 \
            and len(c['static_property']) == 0 and len(c['property']) == 0:
            doc.append('    pass\n\n')
        else:
            doc.append('\n')
    if r['sub_modules'] == []:
        src.append((out_dir + '/' + r['name'] + '.py', ''.join(doc)))
    else:
        src.append((out_dir + '/' + r['name'] + '/' + '__init__.py', ''.join(doc)))
        for sub in r['sub_modules']:
            write_python(sub, out_dir + '/' + r['name'], src)
    return src


def write_zip(src, filename):
    with open(filename, 'wb') as f:
        zf = zipfile.ZipFile(f, 'w')
        for (name, cont) in src:
            zf.writestr(name, cont)
        zf.close()


if __name__ == '__main__':
    import sys, importlib
    args = sys.argv[1:]
    if len(args) == 1:
        args.append('out.zip')
    obj = importlib.import_module(args[0])
    r = parse_module(obj.__name__, obj, recursive=True)
    src = write_python(r, out_dir='.', src=None)
    write_zip(src, args[1])
