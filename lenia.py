import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

SIZE_X = 100
SIZE_Y = 100
SEED = 7
R = 15
T = 10
mu = 0.05
sigma = 0.01
beta = jnp.array([0.9])

key = jax.random.key(SEED)
key, subkey = jax.random.split(key)


def get_polar_radius_matrix(x, y):
    x_axis = jnp.arange(x)
    y_axis = jnp.arange(y)

    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis)
    x_grid = x_grid - x//2
    y_grid = y_grid - y//2

    return jnp.sqrt(x_grid**2 + y_grid**2)

def kernel_core(r, a=4):
    r = jnp.clip(r, 1e-6, 1-1e-6)
    return jnp.exp(a - (a / (4 * r * (1-r))))

def pre_calculate_kernel(beta, dx):
    beta = jnp.array(beta)
    radius = get_polar_radius_matrix(SIZE_X, SIZE_Y) * dx
    radius = jnp.clip(radius, 0, 1 - 1e-6)
    Br = len(beta) * radius
    kernel_shell = beta[jnp.floor(Br).astype(int)] * kernel_core(Br % 1)
    kernel_shell = jnp.where(radius > 1, 0.0, kernel_shell)
    kernel = kernel_shell / jnp.sum(kernel_shell)
    kernel_fft = jnp.fft.fft2(jnp.fft.ifftshift(kernel))
    return kernel_fft

def growth_mapping(potential, mu, sigma):
    return 2*jnp.exp(-(potential-mu)**2/(2*sigma**2)) - 1

@jax.jit
def run_automaton(world, kernel_fft, mu, sigma, dt):
    world_fft = jnp.fft.fft2(world)
    potential_FFT = world_fft * kernel_fft
    potential = jnp.real(jnp.fft.ifft2(potential_FFT))
    growth = growth_mapping(potential, mu, sigma)
    world = world + (dt * growth)
    world = jnp.clip(world, 0 , 1)
    return world


def simulation(key, R, T, mu, sigma, beta, total_time):
    dx = 1/R
    dt = 1/T
    time = 0
    kernel_fft = pre_calculate_kernel(beta, dx)
    # world = jax.random.uniform(key, (SIZE_X, SIZE_Y), jnp.float32)
    world = jnp.zeros((SIZE_X, SIZE_Y))
    # small random blob in the centre
    cx, cy = SIZE_X // 2, SIZE_Y // 2
    world = world.at[cx - 10:cx + 10, cy - 10:cy + 10].set(
        jax.random.uniform(key, (20, 20))
    )
    while time < total_time:
        world = run_automaton(world, kernel_fft, mu, sigma, dt)
        time = time + dt
        plt.imshow(world)
        plt.axis("off")
        plt.pause(0.1)
        plt.clf()


plt.ion()
simulation(key, R, T, mu, sigma, beta, 100)

