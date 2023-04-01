import numpy as np
import matplotlib.pyplot as plt
import random
import datetime


def f(x):
    return np.array([1, x[0], x[1], x[0]**2, x[0] * x[1], x[1]**2, x[0]**3, x[0]**2 * x[1], x[0] * x[1]**2, x[1]**3])


def create_plan(grid, N):
    x = [(random.choice(grid), random.choice(grid)) for _ in range(N)]
    p = [1.0 / N for _ in range(N)]
    return x, p


def create_grid(gridN):
    n = gridN / 2
    return np.arange(-1, 1 + 0.5 / n, 1.0 / n)


def create_grid2D(grid, plan):
    grid2D = [(_x, _y) for _x in grid for _y in grid]
    return [item for item in grid2D if item not in plan[0]]
    # return grid2D  # Если повторные наблюдения разрешены


def find_m(x, p):
    n = len(f((0, 0)))
    M = np.zeros((n, n))
    for i in range(len(x)):
        fx = f(x[i])
        M += p[i] * np.outer(fx, fx.T)
    return M


def d(x_, xj, D):
    return np.dot(np.dot(f(x_).T, D), f(xj))


def calc_delta(x, xj, D, N):
    return 1.0 / N * (d(x, x, D) - d(xj, xj, D)) - \
           1.0 / N**2 * (d(x, x, D) * d(xj, xj, D) - d(x, xj, D)**2)


def show_plan(plan, grid, N, info):
    x = plan[0]
    title = f'plot {len(grid)}-{N} ({info}).png'
    plt.clf()
    plt.title(title)
    for i in range(len(x)):
        plt.scatter(x[i][0], x[i][1])
    plt.savefig(f'output/{title}')


def save_plan(plan, grid, N, info):
    f = open(f'output/plan {len(grid)}-{N}.txt ({info})', 'w+')
    for i in range(len(plan[0])):
        f.write(f'x = ({plan[0][i][0]:+.4f}, {plan[0][i][1]:+.4f})   p = {plan[1][i]:.4f}\n')
    f.close()


def alg_fed(grid, N):
    plan = create_plan(grid, N)
    show_plan(plan, grid, N, 'before')
    save_plan(plan, grid, N, 'before')
    eps = 1e-3
    s = 0
    f = open(f'output/history {len(grid)}-{N}.txt', 'w+')
    while True:
        f.write(f'Iteration {s:3d}.  ')
        print(f'Iteration {s:3d}.  ', end='')
        grid2D = create_grid2D(grid, plan)
        M = find_m(plan[0], plan[1])
        D = np.linalg.inv(M)
        x_s = grid2D[0]
        delta_max = calc_delta(x_s, plan[0][0], D, N)
        i_max = 0
        for i, xj in enumerate(plan[0]):
            for x in grid2D:
                delta = calc_delta(x, xj, D, N)
                if delta > delta_max:
                    delta_max = delta
                    x_s = x
                    i_max = i
        print(f'delta_max = {delta_max:+.8e}   |M| = {np.linalg.det(M):.8e}')
        f.write(f'delta_max = {delta_max:+.8e}   |M| = {np.linalg.det(M):.8e}\n')
        if delta_max < eps:
            print(f'{delta_max:+.8e} < {eps} (delta_max < eps)')
            f.write(f'{delta_max:+.8e} < {eps} (delta_max < eps)\n')
            break
        plan[0][i_max] = x_s
        s += 1
    f.close()
    show_plan(plan, grid, N, 'after')
    save_plan(plan, grid, N, 'after')


if __name__ == '__main__':
    t0 = datetime.datetime.now()
    gridNs = [40]
    Ns = [40]
    for gridN in gridNs:
        for N in Ns:
            alg_fed(create_grid(gridN), N)
    print(datetime.datetime.now() - t0)
