using CairoMakie, IsoPEPS


function plot_TFIM_EvsD()
    x = range(8, 32, 25)
    y = []
    nsites=10
    eigenval,eigenvec=IsoPEPS.eigsolve(IsoPEPS.mat(sum([kron(nsites, i=>(-IsoPEPS.Z),  mod1(i+1, nsites)=>IsoPEPS.Z) for i in 1:nsites])
                                                + sum([-0.5 * kron(nsites, i => IsoPEPS.X) for i in 1:nsites])), 1, :SR; ishermitian=true)
    
    H=Matrix(IsoPEPS.mat(sum([kron(nsites, i=>(-IsoPEPS.Z),  mod1(i+1, nsites)=>IsoPEPS.Z) for i in 1:nsites])
                   + sum([-0.5 * kron(nsites, i => IsoPEPS.X) for i in 1:nsites])))
    eigenmps=vec2mps(Array(eigenvec[1]))
    #mpo=IsoPEPS.transverse_ising_mpo(nsites,0.5)
    mpo=mat2mpo(H)
    new_mps=mps_dot_mpo(eigenmps::MPS,mpo::MPO)
    for i in 1:25
        new_mps1=MPS(IsoPEPS.truncated_bmps(new_mps.tensors,Int(x[i])))
        eigenval1=real(code_dot(eigenmps::MPS,new_mps1::MPS))/real(code_dot(eigenmps::MPS,eigenmps::MPS))
        push!(y,eigenval1)
    end
    fig = Figure()
    ax2 = Axis(fig[1, 1], title = "E vs dmax", xlabel = "dmax", ylabel = "E")
    lines!(ax2, x, y, color = :blue,label="E")
    axislegend(ax2)
    display(fig)
    save("TFIM_EvsD.png", fig)
    fig
end

plot_TFIM_EvsD()

function plot_localX()
    x = range(8, 32, 25)
    y = []  
    nsites=10
    random_mps=generate_mps(32,10)
    mpo=local_X(nsites)
    new_mps=mps_dot_mpo(random_mps::MPS,mpo::MPO)
    for i in 1:25
        new_mps1=MPS(IsoPEPS.truncated_bmps(new_mps.tensors,Int(x[i])))
        expected_X=real(code_dot(random_mps::MPS,new_mps1::MPS))/real(code_dot(random_mps::MPS,random_mps::MPS))
        push!(y,expected_X)
    end
    fig = Figure()
    ax2 = Axis(fig[1, 1], title = "X vs dmax", xlabel = "dmax", ylabel = "X")
    lines!(ax2, x, y, color = :blue,label="X")
    axislegend(ax2)
    display(fig)
    save("XvsD.png", fig)
    fig
end

plot_localX()
