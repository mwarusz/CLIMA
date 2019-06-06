using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

function update!(Q, Qstages, Rstages, Qhat, RKA_explicit, RKA_implicit, dt, is)
  @inbounds @loop for i = (1:length(Q);
                           (blockIdx().x - 1) * blockDim().x + threadIdx().x)
    Qhat[i] = Q[i]
    Qstages[is][i] = 0
    for js = 1:is-1
      commonterm = (RKA_implicit[is, js] - RKA_explicit[is, js]) / RKA_implicit[is, is] * Qstages[js][i]
      Qhat[i] += commonterm + dt * RKA_explicit[is, js] * Rstages[js][i]
      Qstages[is][i] -= commonterm
    end
  end
end
