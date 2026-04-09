module timeevolution

using QuantumToolbox: QuantumObject, Ket, Operator, sesolve, _gen_dimensions
using QuantumToolbox: TimeEvolutionProblem
using OrdinaryDiffEqVerner: Vern7
using SciMLBase: ODEProblem
using LinearAlgebra: mul!, lmul!

"""
    schroedinger_dynamic(tspan, psi0, h; alg=Vern7(), abstol=1e-8, reltol=1e-6, kwargs...)

Integrate the time-dependent Schrodinger equation using QuantumToolbox's `sesolve`.

`h(t, _)` is a function returning a `QuantumObject{Operator}` whose `.data` is
updated in-place (the standard IonSim Hamiltonian convention).

Returns a `QuantumToolbox.TimeEvolutionSol` with fields `.times`, `.states`, `.expect`, etc.
"""
function schroedinger_dynamic(
    tspan,
    psi0::QuantumObject{Ket},
    h::Function;
    alg=Vern7(),
    abstol=1e-8,
    reltol=1e-6,
    kwargs...
)
    H = h(0.0, nothing)
    u0 = copy(psi0.data)
    tlist = collect(Float64, tspan)

    function ode!(du, u, p, t)
        h(t, nothing)
        mul!(du, H.data, u)
        lmul!(-1im, du)
    end

    prob = ODEProblem(ode!, u0, (tlist[1], tlist[end]);
        abstol=abstol, reltol=reltol,
        save_everystep=false, save_end=true, saveat=tlist)

    dims = _gen_dimensions(Ket(), psi0.dims)
    tep = TimeEvolutionProblem(prob, tlist, Ket(), dims, nothing)
    return sesolve(tep, alg; kwargs...)
end

"""
    schroedinger_dynamic(tspan, rho0::QuantumObject{Operator}, h; ...)

Density-matrix form: integrates `dρ/dt = -i[H(t), ρ]` via QuantumToolbox's `sesolve`.
"""
function schroedinger_dynamic(
    tspan,
    rho0::QuantumObject{Operator},
    h::Function;
    alg=Vern7(),
    abstol=1e-8,
    reltol=1e-6,
    kwargs...
)
    H = h(0.0, nothing)
    u0 = copy(rho0.data)
    tlist = collect(Float64, tspan)
    Hρ = similar(u0)

    function ode!(du, u, p, t)
        h(t, nothing)
        mul!(Hρ, H.data, u)
        mul!(du, u, H.data)
        @. du = -1im * (Hρ - du)
    end

    prob = ODEProblem(ode!, u0, (tlist[1], tlist[end]);
        abstol=abstol, reltol=reltol,
        save_everystep=false, save_end=true, saveat=tlist)

    dims = _gen_dimensions(Operator(), rho0.dims)
    tep = TimeEvolutionProblem(prob, tlist, Operator(), dims, nothing)
    return sesolve(tep, alg; kwargs...)
end

end  # module timeevolution
