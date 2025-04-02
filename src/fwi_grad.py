import numpy as np

from devito import Function, norm, mmax, mmin

from examples.seismic import  Receiver
from examples.seismic.acoustic import AcousticWaveSolver



#----------------------------------------compute residual

def compute_residual(res, dobs, dsyn):
    """
    Computes the data residual dsyn - dobs into residual
    """
    if res.grid.distributor.is_parallel:
        # If we run with MPI, we have to compute the residual via an operator
        # First make sure we can take the difference and that receivers are at the
        # same position
        assert np.allclose(dobs.coordinates.data[:], dsyn.coordinates.data)
        assert np.allclose(res.coordinates.data[:], dsyn.coordinates.data)
        # Create a difference operator
        diff_eq = Eq(res, dsyn.subs({dsyn.dimensions[-1]: res.dimensions[-1]}) -
                     dobs.subs({dobs.dimensions[-1]: res.dimensions[-1]}))
        Operator(diff_eq)()
    else:
        # A simple data difference is enough in serial
        res.data[:] = dsyn.data[:] - dobs.data[:]

    return res

#-----------------------------------------calculate gradient

def fwi_gradient(model, geometry, source_locations):
    # Create symbols to hold the gradient
    grad = Function(name="grad", grid=model.grid)
    objective = 0.
    for i in range(len(source_locations)):
        # Create placeholders for the data residual and data
        residual = Receiver(name='residual', grid=model.grid,
                            time_range=geometry.time_axis,
                            coordinates=geometry.rec_positions)
        d_obs = Receiver(name='d_obs', grid=model.grid,
                         time_range=geometry.time_axis,
                         coordinates=geometry.rec_positions)
        d_syn = Receiver(name='d_syn', grid=model.grid,
                         time_range=geometry.time_axis,
                         coordinates=geometry.rec_positions)
        # Update source location
        solver.geometry.src_positions[0, :] = source_locations[i, :]

        # Generate synthetic data from true model
        solver.forward(vp=model.vp, rec=d_obs)

        # Compute smooth data and full forward wavefield u0
        _, u0, _ = solver.forward(vp=model.vp, save=True, rec=d_syn)

        # Compute gradient from data residual and update objective function
        residual = compute_residual(residual, d_obs, d_syn)

        objective += .5*norm(residual)**2
        solver.jacobian_adjoint(rec=residual, u=u0, vp=vp_in, grad=grad)

    return objective, grad
