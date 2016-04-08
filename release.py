#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
release script for SMART-G
'''

from os.path import exists, join, dirname
from os import mkdir, remove, makedirs
from pycuda.compiler import compile

from smartg import dir_src, binnames, dir_bin, src_device, dir_root
from subprocess import Popen, PIPE
from datetime import datetime
import tarfile
import fnmatch


def main():

    version = datetime.now().strftime('%Y%m%d')
    target = join(dir_root, 'release/target/smartg-{}.tar.gz'.format(version))
    skiplist = ['src/*', 'tools/analyze/*', 'makefile*', 'scripts/*',
                'auxdata/*', 'Parametres.txt', 'CHANGELOG.TXT',
                '.gitignore', 'NOTES.TXT', 'TODO.TXT', 'validation/*',
                'release.py', 'notebooks/old_demo_notebook.ipynb',
                '*.hdf', '*.png', '*.ksh']

    if exists(target):
        raise IOError('file {} exists'.format(target))

    # check git status
    if len(Popen(['git', 'diff', '--name-only'], stdout=PIPE).communicate()[0]):
        raise Exception('error, repository is dirty :(')

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
    for f in Popen(['git', 'ls-files'], stdout=PIPE).communicate()[0].split('\n'):

        skip = (len(f) == 0)
        for p in skiplist:
            if fnmatch.fnmatch(f, p):
                skip = True
        if skip:
            # print 'skippd "{}"'.format(f)
            nskip += 1
            continue

        print 'adding "{}"'.format(f)
        tar.add(f, arcname=join('smartg', f))

    print 'skipped {} files'.format(nskip)

    #
    # compilation for pp and sp
    #
    for pp in binnames.keys():
        print 'Compilation in {} mode...'.format({True: 'pp', False: 'sp'}[pp])

        options = []
        if not pp:
            options.append('-DSPHERIQUE')

        options.extend([
            '-gencode', 'arch=compute_20,code=compute_20',
            '-gencode', 'arch=compute_30,code=compute_30',
            '-gencode', 'arch=compute_50,code=compute_50',
            ])

        binary = compile(src_device_content,
                       nvcc='/usr/local/cuda/bin/nvcc',
                       options=options,
                       no_extern_c=True,
                       cache_dir='/tmp/',
                         include_dirs=[dir_src, dir_src+'incRNGs/Random123/',],
                       target='fatbin')

        binname = binnames[pp]
        if exists(binname):
            print 'Warning: removing {}'.format(binname)
            remove(binname)
        with open(binname, 'wb') as f:
            f.write(binary)
            print 'wrote', binname

        assert binname.startswith(dir_root)
        reldir = binname[len(dir_root)+1:]
        tar.add(reldir, arcname=join('smartg', reldir))


    # write git commit in version.txt
    version_file = 'version.txt'
    with open(version_file, 'w') as fp:
        shasum = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE).communicate()[0]
        fp.write(shasum)
        fp.write(version)
        fp.write('\n')
    tar.add(version_file, arcname=join('smartg', version_file))
    tar.close()

    print 'created', target


if __name__ == "__main__":
    main()
