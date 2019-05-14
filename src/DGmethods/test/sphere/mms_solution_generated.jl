const γ_exact = 7//5
const μ_exact = 0
ρ_g(t, x, y, z, r, cπt, sπt) = r*cπt + 3
U_g(t, x, y, z, r, cπt, sπt) = 0.5*(3*x^2 - 1)*cπt
V_g(t, x, y, z, r, cπt, sπt) = 0.5*(3*y^2 - 1)*cπt
W_g(t, x, y, z, r, cπt, sπt) = 0.5*(3*z^2 - 1)*cπt
E_g(t, x, y, z, r, cπt, sπt) = r*cπt + 100
Sρ_g(t, x, y, z, r, cπt, sπt) = 3.0*x*cπt + 3.0*y*cπt + 3.0*z*cπt - pi*r*sπt
SU_g(t, x, y, z, r, cπt, sπt) = (-3.0*x*(3*x^2 - 1)*(r*cπt + 3)*(r^3)*cπt^2 + x*(3*G*(r*cπt + 3) + 2*cπt)*(r*cπt + 3)^2*(r^2) + 0.25*x*(r^2)*((3*x^2 - 1)^2 + (3*y^2 - 1)^2 + (3*z^2 - 1)^2)*cπt^3 + (3*x^2 - 1)*(r*cπt + 3)*(15.0*x + 7.5*y + 7.5*z)*(r^3)*cπt^2 - 1.25*(3*x^2 - 1)*(r^2)*(x*(3*x^2 - 1) + y*(3*y^2 - 1) + z*(3*z^2 - 1))*cπt^3 - (r*cπt + 3)^2*(2*G*x*cπt + 5*pi*(1.5*x^2 - 0.5)*sπt)*(r^3))/(5*(r*cπt + 3)^2*(r^3))
SV_g(t, x, y, z, r, cπt, sπt) = (-3.0*y*(3*y^2 - 1)*(r*cπt + 3)*(r^3)*cπt^2 + y*(3*G*(r*cπt + 3) + 2*cπt)*(r*cπt + 3)^2*(r^2) + 0.25*y*(r^2)*((3*x^2 - 1)^2 + (3*y^2 - 1)^2 + (3*z^2 - 1)^2)*cπt^3 + (3*y^2 - 1)*(r*cπt + 3)*(7.5*x + 15.0*y + 7.5*z)*(r^3)*cπt^2 - 1.25*(3*y^2 - 1)*(r^2)*(x*(3*x^2 - 1) + y*(3*y^2 - 1) + z*(3*z^2 - 1))*cπt^3 - (r*cπt + 3)^2*(2*G*y*cπt + 5*pi*(1.5*y^2 - 0.5)*sπt)*(r^3))/(5*(r*cπt + 3)^2*(r^3))
SW_g(t, x, y, z, r, cπt, sπt) = (-3.0*z*(3*z^2 - 1)*(r*cπt + 3)*(r^3)*cπt^2 + z*(3*G*(r*cπt + 3) + 2*cπt)*(r*cπt + 3)^2*(r^2) + 0.25*z*(r^2)*((3*x^2 - 1)^2 + (3*y^2 - 1)^2 + (3*z^2 - 1)^2)*cπt^3 + (3*z^2 - 1)*(r*cπt + 3)*(7.5*x + 7.5*y + 15.0*z)*(r^3)*cπt^2 - 1.25*(3*z^2 - 1)*(r^2)*(x*(3*x^2 - 1) + y*(3*y^2 - 1) + z*(3*z^2 - 1))*cπt^3 - (r*cπt + 3)^2*(2*G*z*cπt + 5*pi*(1.5*z^2 - 0.5)*sπt)*(r^3))/(5*(r*cπt + 3)^2*(r^3))
SE_g(t, x, y, z, r, cπt, sπt) = (-40*pi*(r*cπt + 3)^3*(r^2)^2*sπt - 2.4*(r*cπt + 3)*(x + y + z)*(r^3)*(2.5*(3*x^2 - 1)^2*cπt^2 + 2.5*(3*y^2 - 1)^2*cπt^2 + 2.5*(3*z^2 - 1)^2*cπt^2 + 10*(r*cπt + 3)*(2*G*(r*cπt + 3)*r - 7*r*cπt - 700))*cπt - 0.5*r*(x*(3*x^2 - 1)*(16*G*(r*cπt + 3)^2*(r^2)*cπt + 24.0*(3*x^2 - 1)*(r*cπt + 3)*(r^2)*cπt^2 + 8*(2*G*(r*cπt + 3) - 7*cπt)*(r*cπt + 3)^2*r - 2.0*r*((3*x^2 - 1)^2 + (3*y^2 - 1)^2 + (3*z^2 - 1)^2)*cπt^3) + y*(3*y^2 - 1)*(16*G*(r*cπt + 3)^2*(r^2)*cπt + 24.0*(3*y^2 - 1)*(r*cπt + 3)*(r^2)*cπt^2 + 8*(2*G*(r*cπt + 3) - 7*cπt)*(r*cπt + 3)^2*r - 2.0*r*((3*x^2 - 1)^2 + (3*y^2 - 1)^2 + (3*z^2 - 1)^2)*cπt^3) + z*(3*z^2 - 1)*(16*G*(r*cπt + 3)^2*(r^2)*cπt + 24.0*(3*z^2 - 1)*(r*cπt + 3)*(r^2)*cπt^2 + 8*(2*G*(r*cπt + 3) - 7*cπt)*(r*cπt + 3)^2*r - 2.0*r*((3*x^2 - 1)^2 + (3*y^2 - 1)^2 + (3*z^2 - 1)^2)*cπt^3))*cπt + 0.4*(r^2)*(x*(3*x^2 - 1) + y*(3*y^2 - 1) + z*(3*z^2 - 1))*(2.5*(3*x^2 - 1)^2*cπt^2 + 2.5*(3*y^2 - 1)^2*cπt^2 + 2.5*(3*z^2 - 1)^2*cπt^2 + 10*(r*cπt + 3)*(2*G*(r*cπt + 3)*r - 7*r*cπt - 700))*cπt^2)/(40*(r*cπt + 3)^3*(r^3))
