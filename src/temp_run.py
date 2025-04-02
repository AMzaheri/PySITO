
for i_iter in range(inpara.n_iter):
    #------------------------------------------------------IDWT loop 
    #for ishot in range(inpara.nshots):
    for ishot in range(nshot_per_rank* rank, nshot_per_rank * (rank + 1)):
        #shift_time_1 = time.time()
        geometry_devito.src_positions[0, :] = source_locations[ishot, :]

        solver = AcousticWaveSolver(modelb, geometry_devito, space_order=4)

        #-------------------------------------------------migration
        cprint(f'[imaging operator]: inversion iter= {i_iter+1}:  source {ishot+1}/{inpara.nshots}', 'magenta')

        image_b = migration_image(modelb, modelb0, geometry_devito, solver)
        imageb =  Laplacian(image_b, modelb0)

        cprint(f'[imaging operator]: inversion iter = {i_iter+1}: source {ishot+1}/{inpara.nshots}', 'magenta')
        image_m = migration_image(modelm, modelb0, geometry_devito, solver)
        imagem = Laplacian(image_m, modelb0) 
        
        #-------------------------------------------------------Warp 
        cprint(f'[Warping Function]: inversion iter = {i_iter+1}: source {ishot+1}/{inpara.nshots}', 'magenta')
        shift = np.zeros(imageb.data.shape)
        nbl = inpara.nbl
        shift[nbl:-nbl, nbl:-nbl], mshift = find_warp(imageb.data[nbl:-nbl, nbl:-nbl],\
                                               imagem.data[nbl:-nbl, nbl:-nbl], inpara)
        #----------------------------------------------------- cost
        cprint(f'[Cost function]: inversion iter = {i_iter+1}', 'magenta')
        cost_val += cost_function(shift, modelb0)

        #------------------------------------------------------alfa
        cprint(f'[Calculating alfa]: inversion iter = {i_iter+1}: source {ishot+1}/{inpara.nshots}', 'magenta')
        alfa = calculate_alfa(imagem, imageb,  modelb0, shift)


        #--------------------------------------------------gradient       
        cprint(f'[Gradient operator]: inversion iter = {i_iter+1}: source {ishot+1}/{inpara.nshots}', 'magenta')
        stime_grad = time.time()

        comm.Barrier()
        grad = compute_id_grad(modelb0, grad_geometry, alfa, inpara, ishot, \
                               receiver_locations_idwt, comm, rank, size)
        gradient_per_source[:,:] += grad.data[:,:]           
        etime_grad = time.time()
        print(f'Calculation of gradient terminated in {etime_grad - stime_grad}')
            
        #-----------------------------------------------collecting partial results
    comm.Barrier()
    #gather partial results and add to the total sum
    comm.Reduce(total_gradient, gradient_per_source, op=MPI.SUM, root=0)
    comm.Reduce(total_cost_val, cost_val, op=MPI.SUM, root=0)
          
    #------------------------------------rank 0
    if rank == 0:
        grad = Function(name="TotalGrad", grid=modelb0.grid)
        grad.data[:,:] =  total_gradient[:,:] / np.max(np.abs(total_gradient))
    
        #-----------------------------------------------velocity update
        cprint(f'[inversion_utils]: inversion iter = {i_iter+1}: updating velocity model', 'magenta')
        update_velocity(modelb0.vp, inpara, grad)
    

        if total_cost_val <= inpara.inv_tol:
            print(f'Total cost is {total_cost_val}: reached the tolerance')
            break
        else:
            print(f'Inversion iteration {i_iter}: Total cost is {total_cost_val}')
            cost_history.append(total_cost_val)

    # Root broadcast updated velocity to others
    comm.bcast(modelb0, root=0) 
    comm.Barrier()   

#-------------------------------------------------------------plots
if rank ==0:
    cprint(f'[Plotting]: writing outputs', 'magenta')

    #gdata = np.max(np.quantile(grad.data[nbl:-nbl,nbl:-nbl], .95))
    gdata = np.max(grad.data)
    #plot_image(grad.data, modelm, inpara.outpath,
    #            'gradient_nrec-%s_nsrc-%s' % (inpara.nrec, inpara.nshots),\
    #              vmin=-.5, vmax=.5, cmap='seismic')

    #------------------------------------------------------------
    np.save(os.path.join(inpara.outpath, 'gradient_arr'), grad.data[nbl:-nbl, nbl:-nbl])
    #--------------------------------------------------------------

    plot_gradient(grad, modelb0, receiver_locations_idwt,\
                   source_locations_idwt, alfa, \
                   'gradient', inpara, vmin=-gdata, vmax=gdata)

    plot_velocity(modelb0, inpara.outpath, 'updated_model')
    
#-----------------------------end of the programme
end_t = time.time()
print(f'Program terminated: duratation: {end_t - start_t}')
