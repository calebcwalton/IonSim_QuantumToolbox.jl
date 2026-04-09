module timeevolution

using QuantumToolbox: QuantumObject, Ket, Operator
using OrdinaryDiffEqVerner: Vern7
using SciMLBase: ODEProblem, solve

"""
    schroedinger_dynamic(tspan, psi0, f; kwargs...)
Integrates the Schrodinger equation where the Hamiltonian `f(t, psi)` is a function of time.
Returns `(tout, states)` where `states` is a vector of QuantumObject{Ket}.

This provides backward compatibility with the QuantumOptics.jl API.
"""
function schroedinger_dynamic(
    tspan,
    psi0::QuantumObject{Ket},
    f::Function;
    fout::Union{Function, Nothing}=nothing,
    kwargs...
)
    u0 = copy(psi0.data)
    tlist = collect(Float64, tspan)

    function ode_func!(du, u, p, t)
        H = f(t, nothing)
        du .= -1im .* (H.data * u)
    end

    prob = ODEProblem(ode_func!, u0, (tlist[1], tlist[end]))
    sol = solve(prob, Vern7(); saveat=tlist, abstol=1e-8, reltol=1e-6, kwargs...)

    states = Vector{typeof(psi0)}(undef, length(sol.t))
    for i in eachindex(sol.t)
        ψ = deepcopy(psi0)
        ψ.data .= sol[i]
        states[i] = ψ
    end
    return sol.t, states
end

function schroedinger_dynamic(
    tspan,
    rho0::QuantumObject{Operator},
    f::Function;
    fout::Union{Function, Nothing}=nothing,
    kwargs...
)
    u0 = copy(rho0.data)
    tlist = collect(Float64, tspan)

    function ode_func!(du, u, p, t)
        H = f(t, nothing)
        du .= -1im .* (H.data * u .- u * H.data)
    end

    prob = ODEProblem(ode_func!, u0, (tlist[1], tlist[end]))
    sol = solve(prob, Vern7(); saveat=tlist, abstol=1e-8, reltol=1e-6, kwargs...)

    states = Vector{typeof(rho0)}(undef, length(sol.t))
    for i in eachindex(sol.t)
        ρ = deepcopy(rho0)
        ρ.data .= sol[i]
        states[i] = ρ
    end
    return sol.t, states
end

end  # module timeevolution
