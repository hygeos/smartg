#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
release script for SMART-G
'''

from __future__ import print_function, absolute_import, division
from os.path import exists, join, dirname, realpath
from os import mkdir, remove, makedirs
from smartg.smartg import dir_src, src_device, dir_root
from smartg.smartg import Smartg
from subprocess import Popen, PIPE
from datetime import datetime
import tarfile
import fnmatch


version = 'v0.9.2'



def main():

    target = join(dir_root, 'release/target/smartg-{}.tar.gz'.format(version))
    tar_root = 'smartg-{}'.format(version)
    skiplist = [
            'NOTES.TXT',
            'TODO.TXT',
            'performance.py',
            'notebooks/Validation_obj3DSurfaceRoughness.ipynb',
            'validation.py',
            'release.py',
            'scripts/*',
            'auxdata/validation/*',
            '.gitmodules',
            'luts',   # will be dealt with separately
            ]

    if exists(target):
        raise IOError('file {} exists'.format(target))
    print('Target is {}'.format(target))

    # check git status
    if len(Popen(['git', 'diff', '--name-only'], stdout=PIPE).communicate()[0]):
        pass
        # raise Exception('error, repository is dirty :(')

    #
    # copy files to the tar
    #
    if not exists(dirname(target)):
        makedirs(dirname(target))
    tar = tarfile.open(target, 'w:gz')
    nskip = 0
    for f in Popen(['git', 'ls-files'], stdout=PIPE).communicate()[0].decode('utf-8').split('\n'):
        if (len(f) == 0) or (True in [fnmatch.fnmatch(f, p) for p in skiplist]):
            # print 'skippd "{}"'.format(f)
            nskip += 1
            continue

        print('adding "{}"'.format(f))
        tar.add(f, arcname=join(tar_root, f))

    # Add luts
    subm = 'luts'
    for f in Popen('cd {} ; git ls-files'.format(subm), stdout=PIPE, shell=True).communicate()[0].decode('utf-8').split('\n'):
        if len(f) == 0:
            continue
        tar.add(join(subm, f), arcname=join(tar_root, subm, f))

    print('skipped {} files'.format(nskip))

    # write git commit in version.txt
    version_file = 'version.txt'
    with open(version_file, 'w') as fp:
        shasum = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE).communicate()[0]
        fp.write(shasum.decode())
        fp.write(version)
        fp.write('\n')
        fp.write(datetime.now().strftime('%Y%m%d:%H%M%S'))
        fp.write('\n')
    tar.add(version_file, arcname=join(tar_root, version_file))
    tar.close()

    print('created', target)


if __name__ == "__main__":
    main()

