"""
Two-ion Molmer-Sorensen entangling gate: Ca40 S1/2 ↔ D5/2 transition.

Gate time: 100 μs, evolution plotted over 400 μs.
Plots SS, DD, SD+DS populations and Bell state fidelity.

Run: julia --project=IonSim_QuantumToolbox.jl ms_gate_entanglement.jl
"""

using IonSim
using IonSim: timeevolution
using IonSim.analytical
using Plots

# --- System setup ---
C1 = Ca40([("S1/2", -1 // 2), ("D5/2", -1 // 2)])
C2 = Ca40([("S1/2", -1 // 2), ("D5/2", -1 // 2)])
L1 = Laser()
L2 = Laser()

chain = LinearChain(
    ions=[C1, C2],
    comfrequencies=(x=3e6, y=3e6, z=1e6),
    selectedmodes=(; z=[1])
)
T = Chamber(iontrap=chain, B=4e-4, Bhat=(x̂ + ẑ) / √2, lasers=[L1, L2])

L1.λ = transitionwavelength(C1, (("S1/2", -1 // 2), ("D5/2", -1 // 2)), T)
L2.λ = transitionwavelength(C1, (("S1/2", -1 // 2), ("D5/2", -1 // 2)), T)
mode = T.iontrap.selectedmodes.z[1]
modecutoff!(mode, 20)

# --- Gate parameters ---
# Target gate time: 100 μs → MS coupling strength = 1/(4 * t_gate) = 2.5 kHz
gate_time = 100.0  # μs
J = 1 / (4 * gate_time * 1e-6)  # MS coupling strength in Hz

# Sideband detuning
ϵ = 40e3  # Hz detuning from motional sideband
d = 80    # AC Stark shift correction

L1.Δ = mode.ν + ϵ - d
L1.k = ẑ
L1.ϵ = x̂

L2.Δ = -mode.ν - ϵ + d
L2.k = ẑ
L2.ϵ = x̂

# Set Rabi frequency: MS coupling J ~ (ηΩ)² / ϵ → Ω = √(J * ϵ) / η
η = abs(lambdicke(mode, C1, L1))
Ω = √(J * ϵ) / η

intensity_from_rabifrequency!(1, Ω, 1, (("S1/2", -1 // 2), ("D5/2", -1 // 2)), T)
intensity_from_rabifrequency!(2, Ω, 1, (("S1/2", -1 // 2), ("D5/2", -1 // 2)), T)

println("System parameters:")
println("  Mode frequency: $(round(mode.ν / 1e6, digits=3)) MHz")
println("  Lamb-Dicke η: $(round(η, digits=4))")
println("  Rabi frequency Ω: $(round(Ω / 1e3, digits=2)) kHz")
println("  Sideband detuning ϵ: $(ϵ / 1e3) kHz")
println("  Target gate time: $(gate_time) μs")
println("  MS coupling J ≈ $(round(J / 1e3, digits=3)) kHz")

# --- Initial state: |SS⟩ ⊗ |n=0⟩ ---
ψ0 = iontensor(ionstate(T, [("S1/2", -1 // 2), ("S1/2", -1 // 2)]), mode[0])

# --- Build Hamiltonian and evolve ---
h = hamiltonian(T, timescale=1e-6, rwa_cutoff=Inf)

tspan = 0:0.25:400  # evolve to 400 μs
println("\nRunning simulation ($(length(tspan)) time steps)...")
@time tout, sol = timeevolution.schroedinger_dynamic(tspan, ψ0, h)

# --- Compute populations ---
SS = real.(expect(ionprojector(T, ("S1/2", -1 // 2), ("S1/2", -1 // 2)), sol))
DD = real.(expect(ionprojector(T, ("D5/2", -1 // 2), ("D5/2", -1 // 2)), sol))
SD = real.(expect(ionprojector(T, ("S1/2", -1 // 2), ("D5/2", -1 // 2)), sol))
DS = real.(expect(ionprojector(T, ("D5/2", -1 // 2), ("S1/2", -1 // 2)), sol))

# Bell state: |Φ+⟩ = (|SS⟩ + i|DD⟩)/√2
# Fidelity with the ideal Bell state = (SS + DD)/2 + Im(⟨SS|ρ|DD⟩)
# For a pure state |ψ⟩, this simplifies using the off-diagonal coherence
SS_proj = ionprojector(T, ("S1/2", -1 // 2), ("S1/2", -1 // 2))
DD_proj = ionprojector(T, ("D5/2", -1 // 2), ("D5/2", -1 // 2))

# Build |SS⟩⟨DD| cross-term projector for coherence measurement
ket_SS = iontensor(ionstate(T, [("S1/2", -1 // 2), ("S1/2", -1 // 2)]), groundstate(mode))
ket_DD = iontensor(ionstate(T, [("D5/2", -1 // 2), ("D5/2", -1 // 2)]), groundstate(mode))
cross = ket_SS * dag(ket_DD)

coherence = expect(cross, sol)
bell_fidelity = @. real((SS + DD) / 2 + imag(coherence))

# --- Print results at gate time ---
gate_idx = findfirst(t -> t >= gate_time, tout)
println("\nAt gate time t = $(tout[gate_idx]) μs:")
println("  P(SS) = $(round(SS[gate_idx], digits=4))")
println("  P(DD) = $(round(DD[gate_idx], digits=4))")
println("  P(SD) = $(round(SD[gate_idx], digits=4))")
println("  P(DS) = $(round(DS[gate_idx], digits=4))")
println("  Bell fidelity = $(round(bell_fidelity[gate_idx], digits=4))")

# --- Plot ---
p1 = plot(
    tout, SS,
    label="SS", lw=2, color=:blue,
    xlabel="Time (μs)", ylabel="Population",
    title="Molmer-Sorensen Gate: Two Ca⁴⁰ Ions",
    legend=:right, size=(900, 500), dpi=150
)
plot!(p1, tout, DD, label="DD", lw=2, color=:red)
plot!(p1, tout, SD .+ DS, label="SD + DS", lw=2, color=:green, ls=:dash)
vline!(p1, [gate_time], label="Gate time ($(gate_time) μs)", color=:black, ls=:dot, lw=1.5)

p2 = plot(
    tout, bell_fidelity,
    label="Bell fidelity", lw=2, color=:purple,
    xlabel="Time (μs)", ylabel="Fidelity",
    title="Bell State Fidelity  |Φ⁺⟩ = (|SS⟩ + i|DD⟩)/√2",
    legend=:topright, size=(900, 300), dpi=150, ylims=(-0.05, 1.05)
)
vline!(p2, [gate_time], label="", color=:black, ls=:dot, lw=1.5)

fig = plot(p1, p2, layout=(2, 1), size=(900, 700), dpi=150)
savefig(fig, "ms_gate_entanglement.png")
println("\nPlot saved to ms_gate_entanglement.png")
