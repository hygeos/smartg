"""Tests for the ray crossings of the Sphere class of the kernel.

The Monte Carlo kernel is not reproducible from one run to the next,
so ``Sphere`` (``src/shapes.h``) is checked where it is deterministic:
a probe kernel runs ``Sphere::IntersectLine`` and ``Sphere::Intersect``
on float32 rays, against numpy references in extended precision on the
same inputs. Two uses meet there: the spherical atmosphere, whose
photons sit on or near a layer boundary of radius 6371 to 6491 km, and
the 3D objects, small spheres seen from far.
"""

from functools import cache
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

EARTH_RADIUS = 6371.0
# one float32 unit in the last place at the radius of the layers
ULP_LAYER = float(np.spacing(np.float32(6400.0)))

_SRC = r"""
#include "communs.h"
#include "shapes.h"

template <typename T>
__device__ void line(const float *o, const float *d, const float *R,
                     float *t, int *hit, int i)
{
    Sphere<T> sphere(static_cast<T>(R[i]));
    Ray<float> ray(make_float3(o[3*i], o[3*i+1], o[3*i+2]),
                   make_float3(d[3*i], d[3*i+1], d[3*i+2]));
    hit[i] = sphere.IntersectLine(ray, &t[2*i], &t[2*i+1]);
}

extern "C" __global__ void crossings(const float *o, const float *d,
                                     const float *R, float *t, int *hit,
                                     int n, int dbl)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (dbl) line<double>(o, d, R, t, hit, i);
    else line<float>(o, d, R, t, hit, i);
}

// a sphere of radius R centred on (cx, cy, cz), turned about x, placed
// by its transformations as the 3D objects are
extern "C" __global__ void intersect(const float *o, const float *d,
                                     float R, float cx, float cy, float cz,
                                     float *t, float *p, float *nn,
                                     int *hit, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Transform<float> tr;
    Transform<float> o2w = tr.Translate(make_float3(cx, cy, cz))
                           * tr.RotateX(30.f);
    Transform<float> w2o = o2w.Inverse(o2w);
    Sphere<float> sphere(&o2w, &w2o, R, -R, R, 360.f);
    Ray<float> ray(make_float3(o[3*i], o[3*i+1], o[3*i+2]),
                   make_float3(d[3*i], d[3*i+1], d[3*i+2]));
    DifferentialGeometry<float> dg;
    float th = 0.f;
    hit[i] = sphere.Intersect(ray, &th, &dg);
    t[i] = th;
    p[3*i] = dg.p.x; p[3*i+1] = dg.p.y; p[3*i+2] = dg.p.z;
    nn[3*i] = dg.nn.x; nn[3*i+1] = dg.nn.y; nn[3*i+2] = dg.nn.z;
}
"""


@cache
def _module() -> Any:
    """Compile the probe kernel once."""
    import pycuda.autoinit  # noqa: F401
    from pycuda.compiler import SourceModule

    from smartg.smartg import DIR_SRC

    return SourceModule(_SRC, no_extern_c=True, include_dirs=[str(DIR_SRC)],
                        options=['-diag-suppress=1444'])


def _f32(x: Any) -> NDArray[np.float32]:
    """An array of float32, as the kernel takes its inputs."""
    return np.asarray(x, dtype=np.float32)


def _flat(v: NDArray[Any]) -> NDArray[np.float32]:
    """Vectors of shape (3, n) as the x, y, z triplets of the kernel."""
    return np.ascontiguousarray(v.T, dtype=np.float32).ravel()


def _crossings(
    o: NDArray[np.float32],
    d: NDArray[np.float32],
    radius: NDArray[np.float32],
    dbl: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    """Run IntersectLine of a sphere centred on the origin.

    Returns t_in, t_out and the hit flags of each ray.
    """
    mod = _module()
    from pycuda.gpuarray import empty, to_gpu

    n = radius.size
    t, hit = empty(2 * n, np.float32), empty(n, np.int32)
    mod.get_function('crossings')(
        to_gpu(_flat(o)), to_gpu(_flat(d)),
        to_gpu(radius.astype(np.float32)), t, hit, np.int32(n),
        np.int32(dbl), block=(256, 1, 1), grid=((n + 255) // 256, 1))
    t = t.get().reshape(n, 2)
    return t[:, 0], t[:, 1], hit.get().astype(bool)


def _reference(
    o: NDArray[np.float32], d: NDArray[np.float32], radius: Any
) -> tuple[NDArray[np.longdouble], NDArray[np.longdouble],
           NDArray[np.longdouble]]:
    """The two crossings in long double, and the closest distance.

    Computed from the float32 inputs themselves, so that only the
    arithmetic of the kernel is judged.
    """
    o, d = o.astype(np.longdouble), d.astype(np.longdouble)
    radius = np.asarray(radius).astype(np.longdouble)
    a = (d * d).sum(0)
    b = (d * o).sum(0)
    c = (o * o).sum(0) - radius * radius
    sq = np.sqrt(np.maximum(b * b - a * c, 0))
    closest = np.sqrt(np.maximum((o * o).sum(0) - b * b / a, 0))
    return (-b - sq) / a, (-b + sq) / a, closest


def _unit(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
    """Random unit vectors, shape (3, n)."""
    u = rng.normal(size=(3, n))
    return u / np.sqrt((u * u).sum(0))


def _shell_rays(
    rng: np.random.Generator, n: int, thickness: float
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32],
           NDArray[np.bool_]]:
    """Photons of a spherical atmosphere and the boundary they go to.

    Half of them sit on a boundary (rounded to float32, as a photon
    moved there), the others inside a layer, with directions down to
    grazing ones. Each goes, as in the move, to the boundary below
    (entering it), to the one above, or back to its own after missing
    the one below (leaving them). Returns the origins, the directions,
    the radii and whether the crossing taken is the entry one.
    """
    u = _unit(rng, n)
    boundary = EARTH_RADIUS + rng.uniform(0, 100, n)
    on = rng.random(n) < 0.5
    r = _f32(np.where(on, boundary, boundary - rng.uniform(0, thickness, n)))
    v = _unit(rng, n)
    # a third of the directions within 6 degrees of the horizontal
    flat = rng.random(n) < 1 / 3
    v[:, flat] -= (v[:, flat] * u[:, flat]).sum(0) * u[:, flat] * \
        rng.uniform(0.9, 1.0, flat.sum())
    v /= np.sqrt((v * v).sum(0))
    down = (v * u).sum(0) < 0
    lower = r - np.float32(thickness)
    meets_lower = (r * r * ((v * u).sum(0) ** 2 - 1) + lower * lower) >= 0
    entry = down & meets_lower
    # going down: the boundary below; going up: the one above; down
    # but missing the one below: back to its own (on a boundary)
    upper = np.where(on, boundary + thickness, boundary)
    own = np.where(on, boundary, upper)
    radius = _f32(np.where(entry, lower, np.where(down, own, upper)))
    return _f32(u * r), _f32(v), radius, entry


def _cos_at(
    o: NDArray[np.float32], d: NDArray[np.float32], t: NDArray[Any]
) -> NDArray[np.longdouble]:
    """Cosine between the ray and the radius at the point at t."""
    o, d = o.astype(np.longdouble), d.astype(np.longdouble)
    p = o + d * np.asarray(t, np.longdouble)
    return np.abs((p * d).sum(0)) / np.sqrt((p * p).sum(0))


@pytest.mark.parametrize("thickness", (1.0, 0.1, 0.01))
def test_shell_crossings_keep_the_float_precision(thickness: float) -> None:
    """The crossing of a layer boundary errs by about one float step.

    The point at the crossing taken by the move lies on the boundary
    within a few float steps at 6400 km (0.49 m), and the distance
    itself is that close where the ray meets the boundary at a fair
    angle. This is the precision of the photon positions: the v2.0.3
    formula of the move erred up to 9 m at a fair angle (median 0.42
    to 0.60 m, against 0.32 m for this one).
    """
    rng = np.random.default_rng(1)
    o, d, radius, entry = _shell_rays(rng, 400_000, thickness)
    t_in, t_out, hit = _crossings(o, d, radius)
    ref_in, ref_out, closest = _reference(o, d, radius)

    # hit or miss agree, except within a few float steps of a tangent
    clear = np.abs(closest - radius) > 8 * ULP_LAYER
    assert np.array_equal(hit[clear], (closest <= radius)[clear])

    met = hit & (closest < radius)
    t = np.where(entry, t_in, t_out)[met]
    ref = np.where(entry, ref_in, ref_out)[met]
    cos = _cos_at(o[:, met], d[:, met], ref)
    err = np.abs(t.astype(np.longdouble) - ref)
    # on the boundary within a few float steps
    assert np.max(err * cos) < 8 * ULP_LAYER
    # the distance itself at a fair angle
    fair = cos > 0.1
    assert np.max(err[fair] * cos[fair]) < 4 * ULP_LAYER
    assert np.median(err[fair]) < ULP_LAYER


def test_shell_miss_gives_the_closest_point() -> None:
    """A line that misses the sphere returns its closest point twice.

    The move takes it as the tangent point of a photon that rounding
    put just outside its upper boundary.
    """
    rng = np.random.default_rng(2)
    n = 10_000
    u = _unit(rng, n)
    r = _f32(EARTH_RADIUS + rng.uniform(0, 100, n))
    v = _unit(rng, n)
    v -= (v * u).sum(0) * u
    v /= np.sqrt((v * v).sum(0))
    # horizontal rays from 10 m to 1 km above a sphere: all miss it
    radius = r - _f32(rng.uniform(0.01, 1.0, n))
    o, d = _f32(u * r), _f32(v)
    t_in, t_out, hit = _crossings(o, d, radius)

    assert not hit.any()
    assert np.array_equal(t_in, t_out)
    tc = -(o.astype(np.float64) * d).sum(0) / (d.astype(np.float64) ** 2
                                                ).sum(0)
    assert np.allclose(t_in, tc, rtol=0, atol=4 * ULP_LAYER)


def test_far_small_spheres_keep_their_precision() -> None:
    """A small sphere seen from far is met within a few float steps.

    The 3D objects are spheres of metres seen from kilometres. There
    the discriminant comes from the point of the line closest to the
    centre, which keeps the radius against the distance.
    """
    rng = np.random.default_rng(3)
    n = 200_000
    radius = _f32(np.exp(rng.uniform(np.log(1e-3), 0.0, n)))
    distance = rng.uniform(10.0, 1000.0, n)
    o = _f32(_unit(rng, n) * distance)
    # aimed within 0.95 radius of the centre
    aim = _unit(rng, n) * radius * np.sqrt(rng.uniform(0, 0.95, n))
    d = aim - o.astype(np.float64)
    d = _f32(d / np.sqrt((d * d).sum(0)))
    t_in, t_out, hit = _crossings(o, d, radius)
    ref_in, ref_out, _ = _reference(o, d, radius)

    assert hit.all()
    step = np.spacing(_f32(distance))
    assert np.all(np.abs(t_in - ref_in) < 8 * step)
    assert np.all(np.abs(t_out - ref_out) < 8 * step)


def test_double_crossings_are_exact_to_the_output() -> None:
    """In double the only error left is the float32 rounding of t."""
    rng = np.random.default_rng(4)
    o, d, radius, entry = _shell_rays(rng, 100_000, 0.1)
    t_in, t_out, hit = _crossings(o, d, radius, dbl=True)
    ref_in, ref_out, closest = _reference(o, d, radius)

    met = hit & (closest < radius)
    t = np.where(entry, t_in, t_out)[met]
    ref = np.where(entry, ref_in, ref_out)[met]
    err = np.abs(t.astype(np.longdouble) - ref)
    assert np.all(err <= np.spacing(np.abs(ref).astype(np.float32)) + 1e-9)


def test_intersect_of_a_placed_sphere() -> None:
    """Intersect of a translated and turned sphere, as an object.

    From outside the ray meets the sphere where it enters it, from
    inside where it leaves it; the point lies on the sphere and the
    normal points outward.
    """
    mod = _module()
    from pycuda.gpuarray import empty, to_gpu

    rng = np.random.default_rng(5)
    n = 20_000
    radius = 0.5
    centre = np.array([3.0, -2.0, 1.0])
    # from 2 to 50 km away, aimed within the sphere, and from inside
    outside = rng.random(n) < 0.8
    start = _unit(rng, n) * np.where(outside, rng.uniform(2, 50, n),
                                     rng.uniform(0, 0.4, n))
    aim = _unit(rng, n) * radius * 0.9 * np.sqrt(rng.random(n))
    d = np.where(outside, aim - start, _unit(rng, n))
    d = _f32(d / np.sqrt((d * d).sum(0)))
    o = _f32(start + centre[:, None])

    t, p, nn, hit = (empty(n, np.float32), empty(3 * n, np.float32),
                     empty(3 * n, np.float32), empty(n, np.int32))
    mod.get_function('intersect')(
        to_gpu(_flat(o)), to_gpu(_flat(d)), np.float32(radius),
        *_f32(centre), t, p, nn, hit, np.int32(n),
        block=(256, 1, 1), grid=((n + 255) // 256, 1))
    t, hit = t.get(), hit.get().astype(bool)
    p = p.get().reshape(n, 3).T.astype(np.float64)
    nn = nn.get().reshape(n, 3).T.astype(np.float64)
    ref_in, ref_out, _ = _reference(
        o.astype(np.float64) - centre[:, None], d, np.float32(radius))

    assert hit.all()
    ref = np.where(outside, ref_in, ref_out)
    step = np.spacing(_f32(np.sqrt((o * o).sum(0)) + radius))
    assert np.all(np.abs(t - ref) < 16 * step)
    rel = p - centre[:, None]
    dist = np.sqrt((rel * rel).sum(0))
    assert np.all(np.abs(dist - radius) < 16 * step)
    assert np.all((nn * rel).sum(0) / dist > 1 - 1e-4)
