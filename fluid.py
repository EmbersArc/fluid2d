import numpy as np
import pyglet as pg
from pyglet.gl import *


GRIDSIZEX = 128
GRIDSIZEY = 256
WINDOW_SCALING_FACTOR = 3.
GAUSS_SEIDEL_TOLERANCE = 1e-4

FPS = 60
VISC = 0.
DIFF = 0.00002
SOURCE = 20000.
FORCE = 150.
DISSOLVE = 0.005


def set_bnd(b: int, x):
    x[0, :] = -x[1, :] if b == 1 else x[1, :]
    x[-1, :] = -x[-2, :] if b == 1 else x[-2, :]
    x[:, 0] = -x[:, 1] if b == 2 else x[:, 1]
    x[:, -1] = -x[:, -2] if b == 2 else x[:, -2]

    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
    x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
    x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])


def lin_solve(b: int, x, x0, ax: float, ay: float, cx: float, cy: float):
    x_last = np.zeros_like(x)
    while True:
        x[1:-1, 1:-1] = (
            x0[1:-1, 1:-1] + (ax * (x[0:-2, 1:-1] + x[2:, 1:-1]) +
                              ay * (x[1:-1, 0:-2] + x[1:-1, 2:]))
        ) / cx
        set_bnd(b, x)

        if np.max(np.abs(x_last-x)) < GAUSS_SEIDEL_TOLERANCE:
            break

        x_last[:, :] = x


def add_source(d, s, dt: float):
    d += dt * s


def diffuse(b: int, x, x0, diff: float, dt: float):
    """
    Spread density across neighboring cells.
    """
    Nx = d.shape[0] - 2
    Ny = d.shape[1] - 2
    ax = dt * diff * Nx * Nx
    ay = dt * diff * Ny * Ny
    lin_solve(b, x, x0, ax, ay, 1. + 4. * ax, 1. + 4. * ay)


def advect(b: int, d, d0, u, v, dt: float):
    """
    Forces density to follow a given velocity field.
    """
    Nx = d.shape[0] - 2
    Ny = d.shape[1] - 2

    row_i = np.arange(1, Nx+1, dtype=np.intp).reshape(-1, 1)
    col_j = np.arange(1, Ny+1, dtype=np.intp).reshape(1, -1)

    x = row_i - dt * Nx * u[1:-1, 1:-1]
    y = col_j - dt * Ny * v[1:-1, 1:-1]

    np.clip(x, 0.5, Nx + 0.5, out=x)
    np.clip(y, 0.5, Ny + 0.5, out=y)

    i0 = x.astype(np.intp)
    j0 = y.astype(np.intp)
    i1 = i0 + 1
    j1 = j0 + 1

    s1 = x - i0
    t1 = y - j0
    s0 = 1. - s1
    t0 = 1. - t1

    d[1:-1, 1:-1] = \
        s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + \
        s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])

    set_bnd(b, d)


def project(u, v, p, div):
    Nx = div.shape[0] - 2
    Ny = div.shape[1] - 2
    hx = 1. / Nx
    hy = 1. / Ny

    div[1:-1, 1:-1] = -0.5 * \
        (hx * (u[2:, 1:-1] - u[0:-2, 1:-1]) +
         hy * (v[1:-1, 2:] - v[1:-1, 0:-2]))

    set_bnd(0, div)

    p[1:-1, 1:-1] = 0.
    set_bnd(0, p)

    lin_solve(0, p, div, 1., 1., 4., 4.)

    u[1:-1, 1:-1] -= 0.5 / hx * (p[2:, 1:-1] - p[0:-2, 1:-1])
    v[1:-1, 1:-1] -= 0.5 / hy * (p[1:-1, 2:] - p[1:-1, 0:-2])

    set_bnd(1, u)
    set_bnd(2, v)


def dens_step(x, x0, u, v, diff: float, dt: float):
    x *= (1. - DISSOLVE)
    add_source(x, x0, dt)
    diffuse(0, x0, x, diff, dt)
    advect(0, x, x0, u, v, dt)


def vel_step(u, v, u0, v0, visc: float, dt: float):
    add_source(u, u0, dt)
    add_source(v, v0, dt)

    diffuse(1, u0, u, visc, dt)
    diffuse(2, v0, v, visc, dt)

    project(u0, v0, u, v)

    advect(1, u, u0, u0, v0, dt)
    advect(2, v, v0, u0, v0, dt)

    project(u, v, u0, v0)


if __name__ == "__main__":
    # Velocity
    v = np.zeros((GRIDSIZEX, GRIDSIZEY), dtype=np.float)
    v_prev = v.copy()
    u = np.zeros((GRIDSIZEX, GRIDSIZEY), dtype=np.float)
    u_prev = u.copy()

    # Density
    d = np.zeros((GRIDSIZEX, GRIDSIZEY), dtype=np.float)
    d_prev = d.copy()

    # Drawing
    WINDOW_SIZEX = int((GRIDSIZEX - 2) * WINDOW_SCALING_FACTOR)
    WINDOW_SIZEY = int((GRIDSIZEY - 2) * WINDOW_SCALING_FACTOR)
    window = pg.window.Window(WINDOW_SIZEX, WINDOW_SIZEY, caption="Fluid")
    window.config.alpha_size = 8
    fps_display = pg.window.FPSDisplay(window=window)

    def screen_coord_to_grid(x, y):
        coords = np.array((x, y), dtype=np.float)
        coords /= WINDOW_SCALING_FACTOR
        coords += 1
        coords = coords.astype(np.intp)
        coords = np.clip(coords, 1, (GRIDSIZEX-2, GRIDSIZEY-2))
        return coords[0], coords[1]

    @window.event
    def on_draw():
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        data = np.zeros(4 * (d.shape[0]-2) * (d.shape[1]-2), dtype=np.uint8)
        data[0::4] = 255
        data[1::4] = 255
        data[2::4] = 0
        data[3::4] = np.minimum(d[1:-1, 1:-1].T, 255).flatten()
        image = pg.image.ImageData(d.shape[0] - 2, d.shape[1] - 2,
                                   'RGBA', data.tobytes())

        window.clear()

        image.blit(x=0, y=0,
                   width=WINDOW_SIZEX,
                   height=WINDOW_SIZEY)

        fps_display.draw()

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, _):
        x, y = screen_coord_to_grid(x, y)

        if buttons & pg.window.mouse.LEFT:
            # add forces
            u_prev[x, y] = FORCE * dx / WINDOW_SCALING_FACTOR
            v_prev[x, y] = FORCE * dy / WINDOW_SCALING_FACTOR

        if buttons & pg.window.mouse.RIGHT:
            # add density
            d_prev[x, y] += SOURCE

    def update(_):
        dt = 1. / FPS
        vel_step(u, v, u_prev, v_prev, VISC, dt)
        dens_step(d, d_prev, u, v, DIFF, dt)
        d_prev.fill(0.)
        u_prev.fill(0.)
        v_prev.fill(0.)

        d_prev[GRIDSIZEX//2, GRIDSIZEY//16] = 10000.
        v_prev[GRIDSIZEX//2, GRIDSIZEY//16] = 50.

    pg.clock.schedule_interval(update, 1/FPS)
    pg.app.run()
