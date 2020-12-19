using LinearAlgebra
using Plots
# plotly()

function swap!(a, b)
    tmp = copy(a)
    a = b
    b = tmp
    return nothing
end

function diffuse!(x, x0, b, config::Grid)
    # a is just a convergence factor
    a = config.dt * config.diff * config.N * config.N
    h = 1

    # Update to use dynamic error checker, instead of static # of iterations
    for k in 1:20
        for i in 2:config.N+1
            for j in 2:config.N+1
                # Gauss-seidel method, main-diagonal dominant (sum of off-diagonal terms needs to be less than diagonal)
                # x[i, j] = (x0[i, j] + a * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1]))/(1+4*a)
                x[i, j] =
                    (
                        x0[i, j] +
                        0.50 * a * (x[i+h, j] + x[i-h, j] + x[i, j+h] + x[i, j-h]) +
                        0.25 * a * (x[i+h, j+h] + x[i-h, j+h] + x[i+h, j-h] + x[i-h, j-h])
                    ) / (1 + 3 * a)
            end
        end
        set_bnd!(x, b, config)
    end
    return nothing
end

function advect!(d, d0, u, v, b, config::Grid)
    dt0 = config.dt * config.N
    for i in 2:config.N+1
        for j in 2:config.N+1
            # Set x and y grid coordinates
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            # Enforce lower and upper bounds to constrain to grid
            i0, i1 = constrain!(x, y, config)
            j0 = floor(Int, y)
            j1 = j0 + 1

            s1 = x - i0
            s0 = 1 - s1
            t1 = y - j0
            t0 = 1 - t1

            d[i, j] =
                s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])

        end
    end
    set_bnd!(d, b, config)
    return nothing
end

function constrain!(x, y, config::Grid{true})
    if x < 1.5
        x = config.N + 1.5
    elseif x > config.N + 1.5
        x = 1.5
    end
    i0 = floor(Int, x) # Take the integer part
    i1 = i0 + 1
    if y < 1.5
        y = config.N + 1.5
    elseif y > config.N + 1.5
        y = 1.5
    end

    return i0, i1
end

function constrain!(x, y, config::Grid{false})
    if x < 1.5
        x = 1.5
    elseif x > config.N + 1.5
        x = N + 1.5
    end
    i0 = floor(Int, x) # Take the integer part
    i1 = i0 + 1
    if y < 1.5
        y = 1.5
    elseif y > config.N + 1.5
        y = config.N + 1.5
    end

    return i0, i1
end

function dens_step!(x, x0, u, v, config::Grid)
    x += config.dt .* x0
    swap!(x0, x) # Swap the names so that x becomes the original, and x0 becomes a guess
    # Rather than swapping and swapping back here, wouldn't it be better to just swap
    # inputs? so rather than x, x0; pass x0, x?
    diffuse!(x, x0, 0, config)
    swap!(x0, x)
    #advect!(x, x0, u, v, 0, config)
    return nothing
end

function vel_step!(u, u0, v, v0, config::Grid)
    u += config.dt .* u0
    swap!(u0, u)
    diffuse!(u, u0, 1, config)
    v += config.dt .* v0
    swap!(v0, v)
    diffuse!(v, v0, 2, config)
    project!(u, v, config)
    swap!(u0, u)
    swap!(v0, v)
    #advect!(u, u0, u0, v0, 1, config)
    #advect!(v, v0, u0, v0, 2, config)
    project!(u, v, config)
    return nothing
end

function project!(u, v, config::Grid)
    p = similar(u)
    div = similar(u)
    h = 1.0 / config.N
    # Calculate divergence using simple gradient
    for i in 2:config.N+1
        for j in 2:config.N+1
            div[i, j] = -0.5 * h * (u[i+1, j] - u[i-1, j] + v[i, j+1] - v[i, j-1])
        end
    end

    set_bnd!(div, 0, config)
    set_bnd!(p, 0, config)

    # Iterate to converge to best approximation
    for k in 1:40
        for i in 2:config.N+1
            for j in 2:config.N+1
                p[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1]) / 4
            end
        end
        set_bnd!(p, 0, config)
    end

    # Subtract off converged divergence from velocity fields to obtain mass-conserving curl
    for i in 2:config.N+1
        for j in 2:config.N+1
            u[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j]) / h
            v[i, j] -= 0.5 * (p[i, j+1] - p[i, j-1]) / h
        end
    end
    set_bnd!(u, 1, config)
    set_bnd!(v, 2, config)
    return nothing
end

function set_bnd!(x, b, config::Grid{false})
    N = config.N
    for i in 2:N+1
        if b == 1 # Do left and right walls
            x[1, i] = -x[2, i]
            x[N+2, i] = -x[N+1, i]
        else
            x[1, i] = x[2, i]
            x[N+2, i] = x[N+1, i]
        end
        if b == 2 # Do top and bottom walls
            x[i, 1] = -x[i, 2]
            x[i, N+2] = -x[i, N+1]
        else
            x[i, 1] = x[i, 2]
            x[i, N+2] = x[i, N+1]
        end

        # Do the corner pieces.
        x[1, 1] = 0.5 * (x[2, 1] + x[1, 2])
        x[1, N+2] = 0.5 * (x[2, N+2] + x[1, N+1])
        x[N+2, 1] = 0.5 * (x[N+1, 1] + x[N+2, 2])
        x[N+2, N+2] = 0.5 * (x[N+1, N+2] + x[N+2, N+1])
    end
    return nothing
end

function set_bnd!(x, b, config::Grid{true})
    N = config.N
    for i in 2:N+1
        if b == 1 # Do top and bottom walls
            x[1, i] = (x[N+1, i] + x[2, i]) / 2
            x[N+2, i] = (x[N+1, i] + x[2, i]) / 2
        else
            x[1, i] = x[2, i]
            x[N+2, i] = x[N+1, i]
        end
        if b == 2 # Do left and right walls
            x[i, 1] = (x[i, 2] + x[i, N+1]) / 2
            x[i, N+2] = (x[i, 2] + x[i, N+1]) / 2
        else
            x[i, 1] = x[i, 2]
            x[i, N+2] = x[i, N+1]
        end

        # Do the corner pieces.
        for i in 1:5
            x[1, 1] = 0.25 * (x[2, 1] + x[1, 2] + x[1, N+2] + x[N+2, 1])
            x[1, N+2] = 0.25 * (x[2, N+2] + x[1, N+1] + x[1, 1] + x[N+2, N+2])
            x[N+2, 1] = 0.25 * (x[N+1, 1] + x[N+2, 2] + x[1, 1] + x[N+2, N+2])
            x[N+2, N+2] = 0.25 * (x[N+1, N+2] + x[N+2, N+1] + x[1, N+2] + x[N+2, 1])
        end
    end

    return nothing
end

mutable struct Grid{P}
    N::Int
    visc::Float64
    diff::Float64
    dt::Float64
end

function Grid(N::Int, periodic::Bool, visc::Float64, diff::Float64, dt::Float64)
    return Grid{periodic}(N, visc, diff, dt)
end

config = Grid(51, true, 0.01, 0.01, 0.01)
##
xSize = config.N + 2
ySize = config.N + 2

u = zeros(xSize, ySize)
v = zeros(xSize, ySize)
dens = zeros(xSize, ySize)

u_prev = zeros(xSize, ySize)
v_prev = zeros(xSize, ySize)
dens_prev = zeros(xSize, ySize)

tc = 1.0 #ARB
emiss = 100
k = 200

tsteps = 200
# for tc = 1:tsteps
anim = @animate for tc in 1:tsteps
    dens_prev[26, 26] = 1000

    u_prev[:, 26] .= cos(2 * pi * tc / k) * emiss
    v_prev[26, :] .= cos(2 * pi * tc / k) * emiss

    vel_step!(u, u_prev, v, v_prev, config)
    dens_step!(dens, dens_prev, u, v, config)
    p1 = heatmap(
        1:xSize,
        1:ySize,
        dens,
        clim = (0, 4),
        aspect_ratio = :equal,
        title = "Density",
    )
    p2 = heatmap(
        1:xSize,
        1:ySize,
        u,
        clim = (-3, 3),
        aspect_ratio = :equal,
        title = "U-velocity",
    )
    p3 = heatmap(
        1:xSize,
        1:ySize,
        v,
        clim = (-3, 3),
        aspect_ratio = :equal,
        title = "V-velocity",
    )
    plot(p1, p2, p3, Layout = (1, 3))
    u_prev = u
    v_prev = v
    dens_prev = dens
end

gif(anim, "dens.gif", fps = 25)
