using LinearAlgebra
using Plots
# plotly()

meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))

mutable struct Grid{P}
    N::Int
    visc::Float64
    diff::Float64
    dt::Float64
end

function Grid(N::Int, periodic::Bool, visc::Float64, diff::Float64, dt::Float64)
    return Grid{periodic}(N, visc, diff, dt)
end

function diffuse!(x, x0, b, diff_rate, config::Grid)
    # a is just a convergence factor
    a = config.dt * diff_rate * config.N * config.N
    h = 1
    err = 1e5
    tol = 1e-12
    # Updated to use dynamic error checker, instead of static # of iterations. Probably not super important.
    while err > tol
        old_x = copy(x)
        for i in 2:config.N+1
            for j in 2:config.N+1
                # Gauss-seidel method, main-diagonal dominant (sum of off-diagonal terms needs to be less than diagonal)
                # x[i, j] = (x0[i, j] + a * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1]))/(1+4*a)

                # This method comes from Jesse, who knows a lot more about stencils than I do. It's a nearest neighbour including diagonals stencil.
                # He tells me this will spread out the error a bit more, which is good.
                x[i, j] =
                    (
                        x0[i, j] +
                        0.50 * a * (x[i+h, j] + x[i-h, j] + x[i, j+h] + x[i, j-h]) +
                        0.25 * a * (x[i+h, j+h] + x[i-h, j+h] + x[i+h, j-h] + x[i-h, j-h])
                    ) / (1 + 3 * a)
            end
        end
        set_bnd!(x, b, config)
        err = abs(sum(x .- old_x))
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
            x, y = constrain(x, y, config)
            i0 = floor(Int, x) # Take the integer part
            i1 = i0 + 1
            j0 = floor(Int, y)
            j1 = j0 + 1

            # Take the fractional part, and use to weight contribution over nearest neighbour squares
            s1 = x % 1
            s0 = 1 - s1
            t1 = y % 1
            t0 = 1 - t1

            # Interpolate where current density resides. Multiplied by 0.99 to ensure convergence.
            # Don't know why it diverges if set to 1. ¯\_(ツ)_/¯
            d[i, j] =
                (
                s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
                ) * 0.99

        end
    end
    set_bnd!(d, b, config)
    return nothing
end

function constrain(x, y, config::Grid{true})
    # For use with periodic boundary conditions.
    # In a while loop, to account for extreme velocities that might be several grids away.
    while x < 1.5 || x > config.N + 1.5
        if x < 1.5
            x = config.N + 1.5 - abs(x)
        elseif x > config.N + 1.5
            x = x - config.N
        end
    end
    while y < 1.5 || y > config.N + 1.5
        if y < 1.5
            y = config.N + 1.5 - abs(y)
        elseif y > config.N + 1.5
            y = y - config.N
        end
    end

    return x, y
end

function constrain(x, y, config::Grid{false})
    # For use with non-periodic boundaries. If fluid would move past the boundary, set to the boundary.
    # Should update to include a boolean-mask, so that obstructions can be included.
    if x < 1.5
        x = 1.5
    elseif x > config.N + 0.5
        x = config.N + 0.5
    end
    if y < 1.5
        y = 1.5
    elseif y > config.N + 0.5
        y = config.N + 0.5
    end

    return x, y
end

function dens_step!(x, x0, u, v, config::Grid)
    x .+= config.dt .* x0 # Take distance difference over a single time step.
    x0, x = x, x0 # Flip names of variables. x0 is overwritten by guesses anyway, so it doesn't matter.
    diffuse!(x, x0, 0, config.diff, config)
    x0, x = x, x0
    advect!(x, x0, u, v, 0, config)
    return nothing
end

function vel_step!(u, u0, v, v0, config::Grid)
    u .+= config.dt .* u0 # Take velocity difference over a single time step.
    u0, u = u, u0 # Flip names of variables.
    diffuse!(u, u0, 1, config.visc, config)
    v .+= config.dt .* v0
    v0, v = v, v0
    diffuse!(v, v0, 2, config.visc, config)
    project!(u, v, config)
    u0, u = u, u0
    v0, v = v, v0
    advect!(u, u0, u0, v0, 1, config)
    advect!(v, v0, u0, v0, 2, config)
    project!(u, v, config)
    return nothing
end

function project!(u, v, config::Grid)
    p = zeros(size(u)) # Was passed into function in original code from paper, not necessary to on modern computers.
    div = zeros(size(u)) # Instantiate divergence matrix.
    h = 1.0 / config.N # Step size. This is actually cancelled out later on, so I'm not entirely sure why it exists.

    # Calculate divergence using simple gradient
    for i in 2:config.N+1
        for j in 2:config.N+1
            div[i, j] = -0.5 * h * (u[i+1, j] - u[i-1, j] + v[i, j+1] - v[i, j-1])
        end
    end

    set_bnd!(div, 0, config)
    set_bnd!(p, 0, config)

    # Tried to use dynamic error here, but it never converged. Static # of steps is the way to go.
    for k in 1:20
        for i in 2:config.N+1
            for j in 2:config.N+1
                # Gauss-seidel method, from the paper.
                # p[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1]) / 4

                # Jesse's stencil, modified to match above method.
                p[i, j] = (
                        div[i, j] +
                        0.50  * (p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1]) +
                        0.25  * (p[i+1, j+1] + p[i-1, j+1] + p[i+1, j-1] + p[i-1, j-1])
                    ) / 3
            end
        end
        set_bnd!(p, 0, config)
    end


    # Subtract off converged calculated divergence from velocity fields to obtain mass-conserving curl. Also introduces pretty swirlies.
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
    # Non-periodic boundary conditions.
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
    # Periodic boundary conditions.
    N = config.N
    for i in 2:N+1

        # Make sure to do the edges correctly, or fluid will only flow from the corner pieces that update, then act as a density source. Not ideal, or correct.
        # Making these the same regardless of whether the velocity is horizontal or vertical. Seems to work.
        x[i, 1] = (x[i, 2] + x[i, N+1]) / 2
        x[i, N+2] = (x[i, 2] + x[i, N+1]) / 2
        x[1, i] = (x[N+1, i] + x[2, i]) / 2
        x[N+2, i] = (x[N+1, i] + x[2, i]) / 2

        # Do the corner pieces. I wasn't entirely sure what to do here, so these are just the averages of all four corner pieces.
        x[1, 1] = 0.25 * (x[2, 1] + x[1, 2] + x[1, N+2] + x[N+2, 1])
        x[1, N+2] = 0.25 * (x[2, N+2] + x[1, N+1] + x[1, 1] + x[N+2, N+2])
        x[N+2, 1] = 0.25 * (x[N+1, 1] + x[N+2, 2] + x[1, 1] + x[N+2, N+2])
        x[N+2, N+2] = 0.25 * (x[N+1, N+2] + x[N+2, N+1] + x[1, N+2] + x[N+2, 1])
    end

    return nothing
end
##
# N, periodic boundaries, diffusion, viscocity, dt
config = Grid(51, false, 0.01, 0.01, 0.01)


##
xSize = config.N + 2
ySize = config.N + 2

x, y = meshgrid(1:xSize, 1:ySize) # Purely for plotting.

# Instantiate velocity components.
u = zeros(xSize, ySize) # Horizontal velocity
v = zeros(xSize, ySize) # Vertical velocity.
u_prev = zeros(xSize, ySize)
v_prev = zeros(xSize, ySize)

# Instantiate density component.
dens = zeros(xSize, ySize)
dens_prev = zeros(xSize, ySize)

# Define center of grid
center = floor(Int, config.N/2)

dens_prev[center - 6 : center + 6, center - 6 : center + 6] .= 10

# Define velocity gradient. Here, I'm using a circular velocity field, with a radius of 1, centered at (x, y) = (floor(N/2), floor(N/2)).

for ic in center - 6:center + 6
    xx = ic
    for jc in center - 6:center + 6
        yy = jc
        u_prev[ic, jc] = (yy-center)*(xx-center)^2 + (yy-center)^2
        v_prev[ic, jc] = -(xx-center)^3 + (yy-center)^2
    end
end
# Scaling the velocity a bit. Higher values imparts a higher initial velocity.
u_prev .*= 10
v_prev .*= 10

# Number of time steps
tsteps = 100

# for tc = 1:tsteps
anim = @animate for tc in 1:tsteps
    global dens_prev, u_prev, v_prev

    l = @layout [a; b c]
    p1 = heatmap(
        1:xSize,
        1:ySize,
        dens,
        clim = (0, 0.2),
        aspect_ratio = :equal,
        # title = "Density",
        legend = false,
        axis = nothing,
        border = :none,
    )
    vel_step!(u, u_prev, v, v_prev, config)
    dens_step!(dens, dens_prev, u, v, config)


    # p2 = heatmap(
    #     1:xSize,
    #     1:ySize,
    #     u,
    #     clim = (-4, 4),
    #     #cbar = false,
    #     aspect_ratio = :equal,
    #     title = "U-velocity",
    # )
    # p3 = heatmap(
    #     1:xSize,
    #     1:ySize,
    #     v,
    #     clim = (-4, 4),
    #     aspect_ratio = :equal,
    #     title = "V-velocity",
    # )
    # p4 = quiver(
    # x,
    # y,
    # quiver = (reshape(u, (1, xSize^2)), reshape(v, (1, ySize^2))), # Be careful here to take into account how arrays work vs how heatmaps are plotted.
    # aspect_ratio = :equal,
    # title = "Velocity field",
    # arrow = arrow(0.01, 0.01)
    # )

    plot(p1, size = (500, 500))
    # , p4, layout = (1, 2)
    u_prev = copy(u)
    v_prev = copy(v)
    dens_prev = copy(dens)
end

gif(anim, "dens.gif", fps = 25)
