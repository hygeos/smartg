#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
release script for SMART-G
'''

from os.path import exists
from os import mkdir, remove
from pycuda.compiler import compile

from smartg import dir_src, binnames, dir_bin, src_device


def main():

    #
    # initialization
    #
    if not exists(dir_bin):
        mkdir(dir_bin)

    src_device_content = open(src_device).read()

    # TODO: check git status
    # write git commit somewhere

    #
    # compilation for pp and sp
    #
    for pp in binnames.keys():
        print 'Compilation in {} mode...'.format({True: 'pp', False: 'sp'}[pp])

        options = ['-DRANDPHILOX4x32_7','-DPROGRESSION']
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
                       include_dirs=[dir_src, dir_src+'incRNGs/Random123/'],
                       target='fatbin')

        binname = binnames[pp]
        if exists(binname):
            print 'Warning: removing {}'.format(binname)
            remove(binname)
        with open(binname, 'wb') as f:
            f.write(binary)
            print 'wrote', binname

if __name__ == "__main__":
    main()
