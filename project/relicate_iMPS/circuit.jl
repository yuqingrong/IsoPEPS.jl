parameter1(p)=rand(3+6p)
function ground_state_circ(params,p)
    nbit = 4
    circ = chain(nbit)
    V(circ,1,2,params,p)
    #V(circ,5,6,params,p)
    for r in 1:p
        U(circ,2,3,params,r)
    end
    for r in 1:p
        U(circ,2,4,params,r)
    end
    #push!(circ, cnot(nbit,1,3))  
    #push!(circ, put(nbit, 3 => H))
    #push!(circ, Measure(nbit; locs=[3,4]))
    return circ
end

function ground_state_circ_x(params,p)
    nbit = 4
    circ = chain(nbit)
    V(circ,1,2,params,p)
    #V(circ,5,6,params,p)
    for r in 1:p
        U(circ,2,3,params,r)
    end
    for r in 1:p
        U(circ,2,4,params,r)
    end
    #push!(circ, cnot(nbit,1,3))  
    push!(circ, put(nbit, 3 => H)) 
    push!(circ, put(nbit, 4 => H)) 
    #push!(circ, Measure(nbit; locs=[3,4]))
    return circ
end

function env_circ1(params,p)
    nbit = 6
    circ = chain(nbit)
    V(circ,1,2,params,p)
    V(circ,4,5,params,p)
    U(circ,2,3,params,p)
    U(circ,5,6,params,p)
    push!(circ, cnot(nbit,6,3))  
    push!(circ, put(nbit, 6 => H))
    #push!(circ, Measure(nbit; locs=[3,6]))
    return circ
end

function env_circ2(params,p)
    nbit = 5
    circ = chain(nbit)
    V(circ,1,2,params,p)
    V(circ,4,5,params,p)
    U(circ,2,3,params,p)
    push!(circ, cnot(nbit,5,3))  
    push!(circ, put(nbit, 5 => H))
    #push!(circ, Measure(nbit; locs=[3,5]))
    return circ
end

function env_circ3(params,p)
    nbit = 4
    circ = chain(nbit)
    V(circ,1,2,params,p)
    V(circ,3,4,params,p)
    push!(circ, cnot(nbit,4,2))  
    push!(circ, put(nbit, 4 => H))
    #push!(circ, Measure(nbit; locs=[2,4]))
    return circ
end

function U(circ,i,j,params,r) 
    nbit = nqubits(circ)
    push!(circ, put(nbit, i => Ry(params[2(r-1)+1])))  
    push!(circ, put(nbit, j => Ry(params[2(r-1)+2])))
    #push!(circ, put(nbit, i => Rx(params[2(r-1)+3])))  
    #push!(circ, put(nbit, j => Rx(params[2(r-1)+4])))
    #push!(circ, put(nbit, i => Rz(params[2(r-1)+5])))  
    #push!(circ, put(nbit, j => Rz(params[2(r-1)+6])))
    push!(circ, cnot(i,j))
    #push!(circ, put(nbit, (i,j) => matblock(sqiswap()))) 
    return circ
end

function V(circ, i, j,params,p)
    nbit = nqubits(circ)
    #expYY
    K(circ,i,j,params[2(p-1)+7])
    push!(circ, put(nbit, i => X))
    K(circ,i,j,-params[2(p-1)+7])
    push!(circ, put(nbit, i => X))
    #expYY(params[2(p-1)+5])
    #push!(circ, put(nbit, (i,j) => matblock(expYY(params[2(p-1)+7]))))
   
    push!(circ, put(nbit, j => Rx(params[2(p-1)+8])))
    push!(circ, put(nbit, j => Rz(params[2(p-1)+9])))

    #push!(circ, cnot(i,j))
    return circ
end


function K(circ,i,j,gamma)
    nbit = nqubits(circ)
    push!(circ, put(nbit, j => Rz(-pi/4)))
    push!(circ, put(nbit, i => Rz(pi/4)))
    push!(circ, cnot(i,j))
    push!(circ, put(nbit, j => Rz(gamma)))
    push!(circ, put(nbit, i => Rz(-gamma)))
    push!(circ, cnot(j,i)) 
    push!(circ, put(nbit, j => Rz(pi/4)))
    push!(circ, put(nbit, i => Rz(-pi/4)))
    return circ
end

function time_evolve_circ(params, p,J,g,dt)
    nbit = 5
    circ = chain(nbit)
    push!(circ, put(nbit, 4 => H))
    push!(circ, cnot(4,5))
    U(circ, 3,4,params,p)
    U(circ, 2,3,params,p)
    TFIM(circ, 3,4, params,p,J,g,dt)
    RightEnv(circ, 1,2,params)
    RightEnv(circ, 4,5,params)
    Daggered(U(circ, 2,3,params,p))
    Daggered(U(circ, 3,4,params,p))
    push!(circ, cnot(4,5))
    push!(circ, put(nbit, 4 => H))
end

function CPHASE(circ, i, j, params,phi,p)     
    nbit = nqubits(circ)
    push!(circ, put(nbit, i => Rz(-phi/2)))
    push!(circ, put(nbit, j => Rz(-phi/2)))
    push!(circ, put(nbit, i => Rx(params[2(p-1)+8]))) #xi_one
    push!(circ, put(nbit, j => Rx(params[2(p-1)+9]))) #xi_two
    push!(circ, put(nbit, (i,j) => matblock(sqiswap()))) 
    push!(circ, put(nbit, i => Rx(-2*params[2(p-1)+10]))) # alpha
    push!(circ, put(nbit, (i,j) => matblock(insqiswap()))) 
    push!(circ, put(nbit, i => Rx(-params[2(p-1)+8])))
    push!(circ, put(nbit, j => Rx(-params[2(p-1)+9])))
end

function TFIM(circ, i, j, params,p,J,g,dt)     
    nbit = nqubits(circ)
    push!(circ, put(nbit, i => Y))
    push!(circ, put(nbit, j => Y))
    K(circ,i,j,J*dt)
    push!(circ, put(nbit, j => X))
    K(circ,i,j,J*dt)
    push!(circ, put(nbit, i => X))
    CPHASE(circ,i,j,params,g*dt,p)
    push!(circ, put(nbit, i => X))
    push!(circ, put(nbit, j => X))
    CPHASE(circ,i,j,params,g*dt,p)
    push!(circ, put(nbit, i => Y))
    push!(circ, put(nbit, j => Y))
    return circ
end

function RightEnv(circ, i, j, params)
    nbit = nqubits(circ)
    push!(circ, put(nbit, j => Rz(params[2(p-1)+5])))
    push!(circ, put(nbit, j => Rx(params[2(p-1)+6])))
    push!(circ, put(nbit, i => Rx(params[2(p-1)+7])))
    #push!(circ, cnot(nbit,j,i))
    push!(circ, put(nbit, i => Rx(-params[2(p-1)+7])))
    push!(circ, put(nbit, j => Rx(-params[2(p-1)+6])))
    push!(circ, put(nbit, j => Rz(-params[2(p-1)+5])))
end

function sqiswap()
    [1 0 0 0; 0 1/sqrt(2) im/sqrt(2) 0; 0 im/sqrt(2) 1/sqrt(2) 0; 0 0 0 1]
end

function insqiswap()
    [1 0 0 0; 0 1/sqrt(2) -im/sqrt(2) 0; 0 -im/sqrt(2) 1/sqrt(2) 0; 0 0 0 1]
end

function expYY(gamma)
    [cos(gamma) 0 0 im*sin(gamma); 0 cos(gamma) -im*sin(gamma) 0; 0 -im*sin(gamma) cos(gamma) 0; im*sin(gamma) 0 0 cos(gamma)]
end

exp_ZZ(J,dt) = exp(-im*(-J*dt*kron(Matrix(Z),Matrix(Z))))

exp_X(h,dt) = exp(-im*(-h*dt*Matrix(X)))

function new_time_evolve_circ(params,p,J,g,dt)
    nbit = 6
    circ = chain(nbit)
    push!(circ, put(nbit, 3 => H))
    push!(circ, cnot(3,2))
    for r in 1:p
        U(circ, 3,4,params,r)
    end
    for r in 1:p
        U(circ, 4,5,params,r)
    end
    RightEnv(circ,1,2,params)
    RightEnv(circ,5,6,params)
    push!(circ, put(nbit,(3,4)=>matblock(exp_ZZ(J,dt))))
    push!(circ, put(nbit,3=>matblock(exp_X(J,dt))))
    push!(circ, put(nbit,4=>matblock(exp_X(J,dt))))
   for r in 1:p
        Daggered(U(circ, 4,5,params,r))
    end
    for r in 1:p
        Daggered(U(circ, 3,4,params,r))
    end
    push!(circ, cnot(3,2))
    push!(circ, put(nbit, 3 => H))
    #push!(circ, Measure(nbit; locs=[1,2,3,4,5,6]))
    return circ
end








