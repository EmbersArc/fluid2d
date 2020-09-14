import numpy as np
import pyglet as pg
from pyglet.gl import *


GRIDSIZE = 256
WINDOW_SCALING_FACTOR = 3.
GAUSS_SEIDEL_TOLERANCE = 1e-4
GAUSS_SEIDEL_ITER = 20
COLOR = (255, 255, 0)

FPS = 60
VISC = 0.
DIFF = 0.00002
SOURCE = 20000.
FORCE = 150.
DISSOLVE = 0.005

MAX_FRAMES = None


def set_bnd(b: int, x):
    x[0, :] = -x[1, :] if b == 1 else x[1, :]
    x[-1, :] = -x[-2, :] if b == 1 else x[-2, :]
    x[:, 0] = -x[:, 1] if b == 2 else x[:, 1]
    x[:, -1] = -x[:, -2] if b == 2 else x[:, -2]

    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
    x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
    x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])


def lin_solve(b: int, x, x0, a: float, c: float):
    x_last = np.zeros_like(x)
    for _ in range(GAUSS_SEIDEL_ITER):
        x[1:-1, 1:-1] = (
            x0[1:-1, 1:-1] + a * (x[0:-2, 1:-1] +
                                  x[2:, 1:-1] +
                                  x[1:-1, 0:-2] +
                                  x[1:-1, 2:])
        ) / c
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
    N = x.shape[0] - 2
    a = dt * diff * N * N
    lin_solve(b, x, x0, a, 1. + 4. * a)


def advect(b: int, d, d0, u, v, dt: float):
    """
    Forces density to follow a given velocity field.
    """
    N = d.shape[0] - 2

    row_i = np.arange(1, N+1, dtype=np.intp).reshape(-1, 1)
    col_j = np.arange(1, N+1, dtype=np.intp).reshape(1, -1)

    dt0 = dt * N
    x = row_i - dt0 * u[1:-1, 1:-1]
    y = col_j - dt0 * v[1:-1, 1:-1]

    np.clip(x, 0.5, N + 0.5, out=x)
    np.clip(y, 0.5, N + 0.5, out=y)

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
    N = div.shape[0] - 2
    h = 1. / N

    div[1:-1, 1:-1] = -0.5 * h * \
        (u[2:, 1:-1] -
         u[0:-2, 1:-1] +
         v[1:-1, 2:] -
         v[1:-1, 0:-2])

    set_bnd(0, div)

    p[1:-1, 1:-1] = 0.
    set_bnd(0, p)

    lin_solve(0, p, div, 1., 4.)

    u[1:-1, 1:-1] -= 0.5 / h * (p[2:, 1:-1] - p[0:-2, 1:-1])
    v[1:-1, 1:-1] -= 0.5 / h * (p[1:-1, 2:] - p[1:-1, 0:-2])

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


class FluidSim(pyglet.window.Window):
    def __init__(self):
        self.WINDOW_SIZE = int((GRIDSIZE-2) * WINDOW_SCALING_FACTOR)
        super(FluidSim, self).__init__(
            self.WINDOW_SIZE, self.WINDOW_SIZE, caption="Fluid")
        # Velocity
        self.v = np.zeros((GRIDSIZE, GRIDSIZE),
                          dtype=np.float)
        self.v_prev = self.v.copy()
        self.u = np.zeros((GRIDSIZE, GRIDSIZE),
                          dtype=np.float)
        self.u_prev = self.u.copy()

        # Density
        self.d = np.zeros((GRIDSIZE, GRIDSIZE),
                          dtype=np.float)
        self.d_prev = self.d.copy()

        self.frame = 0

        # Drawing
        self.config.alpha_size = 8
        self.fps_display = pg.window.FPSDisplay(window=self)

    @staticmethod
    def screen_coord_to_grid(x, y):
        coords = np.array((x, y), dtype=np.float)
        coords /= WINDOW_SCALING_FACTOR
        coords += 1
        coords = coords.astype(np.intp)
        coords = np.clip(coords, 1, GRIDSIZE-2)
        return coords[0], coords[1]

    def on_draw(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        data = np.zeros(4 * (self.d.shape[0]-2)
                        * (self.d.shape[1]-2), dtype=np.uint8)
        data[0::4] = COLOR[0]
        data[1::4] = COLOR[1]
        data[2::4] = COLOR[2]
        data[3::4] = np.minimum(self.d[1:-1, 1:-1].T, 255).flatten()
        image = pg.image.ImageData(self.d.shape[0] - 2, self.d.shape[1] - 2,
                                   'RGBA', data.tobytes())

        self.clear()

        image.blit(x=0, y=0,
                   width=self.WINDOW_SIZE,
                   height=self.WINDOW_SIZE)

        self.fps_display.draw()

    def on_mouse_drag(self, x, y, dx, dy, buttons, _):
        x, y = self.screen_coord_to_grid(x, y)

        if buttons & pg.window.mouse.LEFT:
            # add forces
            self.u_prev[x, y] = FORCE * dx / WINDOW_SCALING_FACTOR
            self.v_prev[x, y] = FORCE * dy / WINDOW_SCALING_FACTOR

        if buttons & pg.window.mouse.RIGHT:
            # add density
            self.d_prev[x, y] += SOURCE

    def update(self, _):
        if self.frame == MAX_FRAMES:
            pg.app.exit()
        else:
            self.frame += 1
        dt = 1. / FPS

        vel_step(self.u, self.v, self.u_prev, self.v_prev, VISC, dt)
        dens_step(self.d, self.d_prev, self.u, self.v, DIFF, dt)

        self.d_prev.fill(0.)
        self.u_prev.fill(0.)
        self.v_prev.fill(0.)

        center = GRIDSIZE//2
        lower = GRIDSIZE//5
        source = 10000
        force = 50

        self.d_prev[lower-1:lower+1, center-1:center+1] = source
        self.u_prev[lower-1:lower+1, center-1:center+1] = force
        self.d_prev[-lower-1:-lower+1, -center-1:-center+1] = source
        self.u_prev[-lower-1:-lower+1, -center-1:-center+1] = -force

    def run(self):
        pg.clock.schedule_interval(self.update, 1/FPS)
        pg.app.run()


if __name__ == "__main__":
    sim = FluidSim()
    sim.run()
