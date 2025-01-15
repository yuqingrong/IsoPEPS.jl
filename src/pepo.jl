mutable struct PEPO{T, AT<:AbstractArray{T,6}}
    tensors::Vector{Vector{AT}}
    function PEPO(tensors::Vector{Vector{AT}}) where {T, AT<:AbstractArray{T,6}}
        Ly=length(tensors)
        Lx=length(tensors[1])
        physical_dim=size(tensors[1][1],5)
        @assert all(size(tensors[y][x],5)==size(tensors[y][x],6)==physical_dim for y in 1:Ly, x in 1:Lx)
        @assert all(size(tensors[y][x],4)==size(tensors[y][mod1(x+1,Lx)],1) for y in 1:Ly, x in 1:Lx)
        @assert all(size(tensors[y][x],3)==size(tensors[mod1(y+1,Ly)][x],2) for y in 1:Ly, x in 1:Lx)
        new{T,AT}(tensors)
    end
end


