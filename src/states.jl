
qubit(theta, phi) = PhysicalSite(reshape([cos(theta), exp(phi * im) * sin(theta)], (1, 2, 1)), false)

