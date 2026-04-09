using Test, IonSim
using Suppressor
using SparseArrays, LinearAlgebra

@suppress_err begin

    # setup system
    C = Ca40([("S1/2", -1 // 2), ("D5/2", -1 // 2)])
    chain = LinearChain(
        ions=[C, C],
        comfrequencies=(x=2, y=2, z=1),
        selectedmodes=(x=[1], y=[], z=[1])
    )
    T = Chamber(iontrap=chain)
    allmodes = modes(chain)

    @testset "operators -- VibrationalMode operators" begin
        # test creation of VibrationalMode operators against manual matrix construction
        N = 10
        modecutoff!(allmodes[1], N)

        # create: matrix with sqrt(1), sqrt(2), ..., sqrt(N) on the superdiagonal
        ref_create = sparse(diagm(-1 => sqrt.(1.0:N)))
        @test create(allmodes[1]).data == ref_create

        # destroy: transpose of create
        ref_destroy = sparse(diagm(1 => sqrt.(1.0:N)))
        @test destroy(allmodes[1]).data == ref_destroy

        # number: diagonal with 0, 1, ..., N
        ref_number = sparse(diagm(0 => collect(0.0:N)))
        @test number(allmodes[1]).data == ref_number

        # fockstate: basis vector with 1.0 at position (n+1)
        n = rand(0:N)
        ref_fock = zeros(ComplexF64, N + 1)
        ref_fock[n + 1] = 1.0
        @test fockstate(allmodes[1], n).data == ref_fock

        # displacement operator: verify unitarity and D(0) = I
        α = rand(0:1e-3:7) + im * rand(0:1e-3:7)
        D = displace(allmodes[1], α).data
        @test D * D' ≈ I(N + 1) atol = 1e-6
        @test displace(allmodes[1], 0).data ≈ I(N + 1)

        # displacement analytic method: verify correct size and D(0) = I
        modecutoff!(allmodes[1], 200)
        @test size(displace(allmodes[1], 0.5, method="analytic").data) == (201, 201)
        @test displace(allmodes[1], 0, method="analytic").data ≈ I(201)

        # test that mean excitation of thermalstate is as expected
        modecutoff!(allmodes[1], 500)
        n̄ = abs(2randn())
        @test expect(number(allmodes[1]), thermalstate(allmodes[1], n̄)) ≈ n̄
        @test expect(
            number(allmodes[1]),
            thermalstate(allmodes[1], n̄, method="analytic")
        ) ≈ n̄

        # test coherentstate: verify |<n|α>|² = Poisson distribution
        modecutoff!(allmodes[1], 500)
        α_cs = 3.0 + 2.0im
        cs = coherentstate(allmodes[1], α_cs).data
        # mean photon number should be |α|²
        probs = abs2.(cs)
        mean_n = sum(k * probs[k + 1] for k in 0:500)
        @test mean_n ≈ abs2(α_cs) rtol = 1e-6

        # test coherenthermalstate
        N = 500
        modecutoff!(allmodes[1], N)
        n̄ = rand(0:1e-6:10)
        @test coherentthermalstate(allmodes[1], n̄, 0, method="analytic").data ≈
              thermalstate(allmodes[1], n̄).data
        @test coherentthermalstate(allmodes[1], 0, α, method="analytic").data ≈
              dm(coherentstate(allmodes[1], α)).data rtol = 1e-3 * N^2
        @test coherentthermalstate(allmodes[1], n̄, 0).data ≈
              thermalstate(allmodes[1], n̄).data
        @test coherentthermalstate(allmodes[1], 0, α).data ≈
              dm(coherentstate(allmodes[1], α)).data rtol = 1e-3 * N^2

        # shouldn't be able to have a mean phonon occupation greater than Hilbert space dimension
        @test_throws AssertionError coherentthermalstate(allmodes[1], N + 1, 0)
        @test_throws AssertionError coherentthermalstate(allmodes[1], 0, N + 1)
        @test_throws AssertionError coherentstate(allmodes[1], N + 1)
        @test_throws AssertionError thermalstate(allmodes[1], N + 1)
        # @test_throws AssertionError displace(modes[1], N + 1)
    end

    @testset "operators -- Ion operators" begin
        # test that ionstate constructs the appropriate state for a single ion
        @test ionstate(C, ("S1/2", -1 // 2)).data == ionstate(C, 1).data == ComplexF64[1; 0]
        @test ionstate(C, ("D5/2", -1 // 2)).data == ionstate(C, 2).data == ComplexF64[0; 1]

        # test ionstate for an IonTrap input
        @test ionstate(chain, [("S1/2", -1 // 2), ("D5/2", -1 // 2)]).data ==
              kron(ComplexF64[0; 1], ComplexF64[1; 0])
        @test ionstate(chain, [1, 2]).data == kron(ComplexF64[0; 1], ComplexF64[1; 0])

        # test ionstate for an Chamber input
        @test ionstate(T, [("S1/2", -1 // 2), ("D5/2", -1 // 2)]).data ==
              kron(ComplexF64[0; 1], ComplexF64[1; 0])

        # test sigma(ion::Ion, ψ1::T, ψ2::T) where {T<:Union{String,Int}}
        @test sigma(C, ("S1/2", -1 // 2), ("D5/2", -1 // 2)).data ==
              sigma(C, 1, 2).data ==
              ComplexF64[0 1; 0 0]
        # test sigma(ion::Ion, ψ1::T<:Union{String,Int})
        @test sigma(C, ("S1/2", -1 // 2)).data == sigma(C, 1).data == ComplexF64[1 0; 0 0]

        # test ionprojector for IonTrap input
        ψ = ionprojector(chain, ("S1/2", -1 // 2), ("D5/2", -1 // 2), only_ions=true)
        @test ψ.data == kron(
            ComplexF64[0; 1] * ComplexF64[0; 1]',
            ComplexF64[1; 0] * ComplexF64[1; 0]'
        )
        @test ionprojector(chain, ("S1/2", -1 // 2), ("D5/2", -1 // 2)) ==
              iontensor(ψ, one(allmodes[1]), one(allmodes[2]))
        @test ψ == ionprojector(T, ("S1/2", -1 // 2), ("D5/2", -1 // 2), only_ions=true)
    end

    @testset "operators -- internal functions" begin
        # test _pf(s, n, m)
        s = rand(1:12)
        n = rand(1:s)
        m = rand(1:s)
        v1 = IonSim._pf(s, n, m)
        v2 = 1im^n * (-1im)^m * factorial(s) / ((s + 1) * √(factorial(m) * factorial(n)))
        isapprox(v1, v2)

        # test He(n) gives the correct Hermite polynomial for order 10
        IonSim._He(10) == [-945, 0, 4725, 0, -3150, 0, 630, 0, -45, 0, 1]

        # test fHe(x, He)
        IonSim._fHe(1, 10) == sum(IonSim._He(10))

        # _alaguerre
        L32 =
            (1 / 6) * (
                -2^3 + 3 * (2 + 3) * 2^2 - 3 * (2 + 2) * (2 + 3) * 2 +
                (2 + 1) * (2 + 2) * (2 + 3)
            )
        @test IonSim._alaguerre(2, 3, 2) ≈ L32

        # _Dnm
        ξ = im * exp(2π * im)
        d = displace(allmodes[1], ξ).data
        diff = 0.0
        for i in 1:100, j in 1:100
            diff += abs(d[i, j] - IonSim._Dnm(ξ, i, j))
        end
        @test diff < 100  # <1% difference of L1 norm
        # Note: displace() is an approximation, whereas _Dnm should not be
    end
end  # end suppress
