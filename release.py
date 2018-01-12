#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
release script for SMART-G
'''

from __future__ import print_function, absolute_import, division
from os.path import exists, join, dirname, realpath
from os import mkdir, remove, makedirs
from pycuda.compiler import compile
from smartg.smartg import dir_src, binnames, dir_bin, src_device, dir_root
from smartg.smartg import Smartg
from subprocess import Popen, PIPE
from datetime import datetime
import tarfile
import fnmatch


def main():

    version = datetime.now().strftime('%Y%m%d')
    target = join(dir_root, 'release/target/smartg-{}.tar.gz'.format(version))
    skiplist = [
            'NOTES.TXT',
            'TODO.TXT',
            'performance.py',
            'smartg/src/*',
            'auxdata/validation/*',
            'smartg/tools/water/*',
            'notebooks/demo_notebook_ALIS.ipynb',
            'validation.py',
            '.gitignore',
            'release.py',
            ]

    if exists(target):
        raise IOError('file {} exists'.format(target))
    print('Target is {}'.format(target))

    # check git status
    if len(Popen(['git', 'diff', '--name-only'], stdout=PIPE).communicate()[0]):
        pass
        # raise Exception('error, repository is dirty :(')

    #
    # initialization
    #
    if not exists(dir_bin):
        mkdir(dir_bin)

    src_device_content = open(src_device).read()

    #
    # copy files to the tar
    #
    if not exists(dirname(target)):
        makedirs(dirname(target))
    tar = tarfile.open(target, 'w:gz')
    nskip = 0
    for f in Popen(['git', 'ls-files'], stdout=PIPE).communicate()[0].decode('utf-8').split('\n'):

        skip = (len(f) == 0)
        for p in skiplist:
            if fnmatch.fnmatch(f, p):
                skip = True
        if skip:
            # print 'skippd "{}"'.format(f)
            nskip += 1
            continue

        print('adding "{}"'.format(f))
        tar.add(f, arcname=join('smartg', f))

    print('skipped {} files'.format(nskip))

    #
    # compilation for pp and sp
    #
    for binopts in binnames.keys():
        print('Compilation binary {} with options {} ...'.format(binnames[binopts], binopts))

        options = []
        if not binopts[0]:
            options.append('-DSPHERIQUE')
        if binopts[1]:
            options.append('-DALIS')
        options.append('-DPHILOX')

        options.extend([
            '-gencode', 'arch=compute_20,code=compute_20',
            '-gencode', 'arch=compute_30,code=compute_30',
            '-gencode', 'arch=compute_50,code=compute_50',
            '-gencode', 'arch=compute_60,code=compute_60',
            ])

        binary = compile(src_device_content,
                       nvcc='nvcc',
                       options=options,
                       no_extern_c=True,
                       cache_dir='/tmp/',
                         include_dirs=[dir_src, dir_src+'incRNGs/Random123/',],
                       target='fatbin')

        binname = binnames[binopts]
        if exists(binname):
            print('Warning: removing {}'.format(binname))
            remove(binname)
        with open(binname, 'wb') as f:
            f.write(binary)
            print('wrote', binname)

        assert binname.startswith(dir_root)
        reldir = binname[len(dir_root)+1:]
        tar.add(reldir, arcname=join('smartg', reldir))


    # write git commit in version.txt
    version_file = 'version.txt'
    with open(version_file, 'w') as fp:
        shasum = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE).communicate()[0]
        fp.write(shasum.decode())
        fp.write(version)
        fp.write('\n')
    tar.add(version_file, arcname=join('smartg', version_file))
    tar.close()

    print('created', target)


if __name__ == "__main__":
    main()

