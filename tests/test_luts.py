#!/usr/bin/env python
# -*- coding: utf-8 -*-


from nose.tools import raises
import numpy as np
from tools.luts import LUT, MLUT, read_mlut_hdf, merge
import os

def create_mlut():
    np.random.seed(0)
    m = MLUT()
    m.add_axis('a', np.linspace(100, 150, 5))
    m.add_axis('b', np.linspace(5, 8, 6))
    m.add_axis('c', np.linspace(0, 1, 7))
    m.add_dataset('data1', np.arange(5*6, dtype='float').reshape(5,6), ['a', 'b'])
    m.add_dataset('data2', np.random.randn(5, 6, 7), ['a', 'b', 'c'])
    m.add_dataset('data3', np.random.randn(10, 12))
    m.set_attr('x', 12)   # set MLUT attributes
    m.set_attrs({'y':15, 'z':8})

    return m

def create_lut():
    z = np.linspace(0, 120., 80)
    P0 = np.linspace(980, 1030, 6)
    Pdata = P0.reshape(1,-1)*np.exp(-z.reshape(-1,1)/8) # dimensions (z, P0)
    return LUT(Pdata, axes=[z, P0], names=['z', 'P0'])

@raises(Exception)
def test_getlut1():
    m = create_mlut()
    m['data4']

def test_getlut2():
    m = create_mlut()
    m['data1']

def test_operations1():
    def check_operations1(fn, result):
        m0 = create_mlut()
        m0.set_attr('z', 5)
        m1 = create_mlut()
        m1['data1'].data[:] = 2

        m = fn(m0, m1)
        assert m.attrs['x'] == 12
        assert not 'z' in m.attrs
        assert 'a' in m.axes
        assert 'b' in m.axes
        assert 'c' in m.axes
        assert 'data1' in m.datasets()
        assert 'data2' in m.datasets()
        assert 'data3' in m.datasets()

        assert m['data1'][1,1] == result

        # check that same result is obtained through MLUT and LUT operation
        assert np.allclose(fn(m0, m1)['data2'][:,:,:], fn(m0['data2'], m1['data2'])[:,:,:])

    for (op, res) in [
            (lambda x, y: x+y, 9.),
            (lambda x, y: x-y, 5.),
            (lambda x, y: y-x, -5.),
            (lambda x, y: x*y, 14.),
            (lambda x, y: x/y, 3.5),
            ]:
        yield check_operations1, op, res


@raises(AssertionError)
def test_operations2():
    # Operations should not be allowed between inconsistent MLUTS
    m0 = create_mlut()
    m0.add_dataset('data4', np.random.randn(10, 12))
    m1 = create_mlut()
    m0 + m1

def test_operations3():
    def check_operations3(fn, result):
        m0 = create_mlut()

        # operate on the MLUT
        assert fn(m0)['data1'][1,1] == res

        # operate on the LUT
        assert fn(m0['data1'])[1,1] == res

        # operate on the array
        assert fn(m0['data1'][1,1]) == res

    for (op, res) in [
            (lambda x: x+2, 9.),
            (lambda x: 2+x, 9.),
            (lambda x: x-2, 5.),
            (lambda x: 2-x, -5.),
            (lambda x: x*2, 14.),
            (lambda x: 2*x, 14.),
            (lambda x: x/2, 3.5),
            (lambda x: 2./(x+1), 0.25),
            ]:
        yield check_operations3, op, res

def test_indexing():
    m = create_mlut()
    for i, d in enumerate(m.datasets()):
        if m[d].ndim == 2:
            assert np.allclose(m[d][:,:], m[i][:,:])
        elif m[d].ndim == 3:
            assert np.allclose(m[d][:,:,:], m[i][:,:,:])


def test_merge():
    mluts = []
    for p1 in np.arange(5):
        for p2 in np.arange(3):
            m = create_mlut()
            m.set_attr('p1', p1)
            m.set_attr('p2', p2)
            mluts.append(m)
    m = merge(mluts, ['p1', 'p2'])

    assert len(m.datasets()) == 3
    assert m[0].shape == (5, 3, 5, 6)
    assert 'x' in m.attrs


def test_merge2():
    # test equality between merging luts and mluts
    mluts = []
    for p1 in np.arange(5):
        for p2 in np.arange(3):
            m = create_mlut()
            m.set_attr('p1', p1)
            m.set_attr('p2', p2)
            mluts.append(m)
    m = merge(mluts, ['p1', 'p2'])

    assert m[0] == merge(map(lambda x: x[0], mluts), ['p1', 'p2'])

def test_equality():
    m0 = create_mlut()
    m1 = create_mlut()

    assert m0 == m1
    assert m0 != 2


def test_convert():
    m = MLUT()
    m.add_axis('a', np.linspace(100, 150, 5))
    m.add_axis('b', np.linspace(5, 8, 6))
    m.add_dataset('data1', np.arange(5*6, dtype='float').reshape(5,6), ['a', 'b'])
    m.set_attr('x', 12)   # set MLUT attributes
    m.set_attrs({'y':15, 'z':8})

    assert m == m[0].to_mlut()

def test_write_read_mlut():
    # write a mlut, read it again, should be equal
    import tempfile
    m0 = create_mlut()
    tmpdir = tempfile.mkdtemp()
    filename = os.path.join(tmpdir, 'mlut.hdf')
    try:
        m0.save(filename)
        m1 = read_mlut_hdf(filename)
    except:
        raise
    finally:
        # always remove that file
        os.remove(filename)
        os.rmdir(tmpdir)

    assert m0.equal(m1, strict=True, show_diff=True)


def test_write_read_mlut2():
    # partial read of mlut
    import tempfile
    m0 = create_mlut()
    for d in m0.datasets():
        tmpdir = tempfile.mkdtemp()
        filename = os.path.join(tmpdir, 'mlut.hdf')
        try:
            m0.save(filename)
            m1 = read_mlut_hdf(filename, datasets=[d])
        except:
            raise
        finally:
            # always remove that file
            os.remove(filename)
            os.rmdir(tmpdir)

        assert m0[d] == m1[0]

