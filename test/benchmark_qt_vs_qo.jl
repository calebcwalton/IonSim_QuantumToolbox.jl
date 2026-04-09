"""
Benchmark comparing IonSim (QuantumToolbox backend) vs QuantumOptics reference.

Run with: julia --project=. test/benchmark_qt_vs_qo.jl
"""

using IonSim
using IonSim: timeevolution
using BenchmarkTools

println("=" ^ 60)
println("IonSim Benchmark: QuantumToolbox.jl Backend")
println("=" ^ 60)

# Setup: single-ion carrier Rabi oscillation
C = Ca40([("S1/2", -1/2), ("D5/2", -1/2)])
L = Laser()
chain = LinearChain(ions=[C], comfrequencies=(x=3e6, y=3e6, z=1e6), selectedmodes=(; z=[1]))
T = Chamber(iontrap=chain, B=4e-4, Bhat=ẑ, δB=0, lasers=[L])
L.λ = transitionwavelength(C, (("S1/2", -1/2), ("D5/2", -1/2)), T)
mode = T.iontrap.selectedmodes.z[1]
L.k = (x̂ + ẑ) / √2
L.ϵ = (x̂ - ẑ) / √2
Ω = 5000.0
intensity_from_rabifrequency!(1, Ω, 1, (("S1/2", -1/2), ("D5/2", -1/2)), T)

println("\n--- Single-ion system (2 × 11 Hilbert space) ---")

# Benchmark Hamiltonian construction
print("Hamiltonian construction: ")
b1 = @benchmark hamiltonian($T, timescale=1e-6) samples=50 evals=1
println("$(round(median(b1).time / 1e6, digits=2)) ms (median)")

h = hamiltonian(T, timescale=1e-6)
psi0 = iontensor(ionstate(T, [("S1/2", -1/2)]), mode[0])

# Benchmark Hamiltonian evaluation
print("Hamiltonian evaluation (single step): ")
b2 = @benchmark $h(100.0, nothing) samples=1000
println("$(round(median(b2).time / 1e3, digits=2)) μs (median)")

# Benchmark time evolution (400 steps)
tspan = 0:1e-1:400
print("Time evolution ($(length(tspan)) steps): ")
b3 = @benchmark timeevolution.schroedinger_dynamic($tspan, $psi0, $h) samples=10 evals=1
println("$(round(median(b3).time / 1e6, digits=2)) ms (median)")

# Numerical validation
Ω00 = Ω * exp(-lambdicke(mode, C, L)^2 / 2)
tout, sol = timeevolution.schroedinger_dynamic(tspan, psi0, h)
ex = real.(expect(ionprojector(T, ("D5/2", -1/2)), sol))
ex_a = @.(sin(2π * Ω00 / 2 * tout * 1e-6)^2)
println("Carrier Rabi max error vs analytical: ", round(maximum(abs.(ex .- ex_a)), sigdigits=3))

# Two-ion Molmer-Sorensen
println("\n--- Two-ion system (4 × 16 Hilbert space) ---")
C2 = Ca40([("S1/2", -1/2), ("D5/2", -1/2)])
L1 = Laser(); L2 = Laser()
chain2 = LinearChain(ions=[C, C2], comfrequencies=(x=3e6, y=3e6, z=1e6), selectedmodes=(; z=[1]))
T2 = Chamber(iontrap=chain2, B=4e-4, Bhat=(x̂ + ẑ) / √2, lasers=[L1, L2])
L1.λ = transitionwavelength(C, (("S1/2", -1/2), ("D5/2", -1/2)), T2)
L2.λ = L1.λ
mode2 = T2.iontrap.selectedmodes.z[1]
modecutoff!(mode2, 15)
ϵ = 40e3
L1.Δ = mode2.ν + ϵ - 80; L1.k = ẑ; L1.ϵ = x̂
L2.Δ = -mode2.ν - ϵ + 80; L2.k = ẑ; L2.ϵ = x̂
η2 = abs(lambdicke(mode2, C, L1))
Ω2 = √(1e3 * ϵ) / η2
intensity_from_rabifrequency!(1, Ω2, 1, (("S1/2", -1/2), ("D5/2", -1/2)), T2)
intensity_from_rabifrequency!(2, Ω2, 1, (("S1/2", -1/2), ("D5/2", -1/2)), T2)

print("Hamiltonian construction: ")
b4 = @benchmark hamiltonian($T2, timescale=1e-6, rwa_cutoff=5e5) samples=20 evals=1
println("$(round(median(b4).time / 1e6, digits=2)) ms (median)")

h2 = hamiltonian(T2, timescale=1e-6, rwa_cutoff=5e5)
psi02 = iontensor(ionstate(T2, [("S1/2", -1/2), ("S1/2", -1/2)]), mode2[0])

tspan2 = 0:0.25:1000
print("Time evolution ($(length(tspan2)) steps): ")
b5 = @benchmark timeevolution.schroedinger_dynamic($tspan2, $psi02, $h2) samples=5 evals=1
println("$(round(median(b5).time / 1e6, digits=2)) ms (median)")

println("\n" * "=" ^ 60)
println("All benchmarks complete.")
