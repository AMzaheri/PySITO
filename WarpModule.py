import sys
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, lsqr
import numpy as np
from numpy.random import rand
#-----------------------------------------------

global mGrad, eps, nx, nz

#nx = 10; nz = 7
nz = 241 ; nx = 602
eps = [.01, 0, 0]
mGrad = np.random.rand(nx*nz)
#x = rand(nx*nz,1); y = rand(5*nx*nz,1); mGrad=rand(nx*nz,1)
#-----------------------------------------
#-------------------------------------------- total_op_forward

def total_op_for(x):
    #print(f'nx = {nx}, nz= {nz}')
    ne = nx * nz
    y = np.zeros(5*ne)
    y[0: ne] = warping_op_for(x) #Warp term
    #print(f'total_op_for: x.shape= {x.shape}, y[ne: 2*ne].shape = {y[ne: 2*ne].shape}')
    y[ne: 2*ne] = eps[0] * x.flatten()  # 0th order Tikhonov
    y[2*ne: 3*ne] = eps[1] * derivative1_op_x_for(x, nx, nz) # 1st order Tikhonov (x)
    y[3*ne: 4*ne] = eps[1] * derivative1_op_z_for(x, nx, nz) # 1st order Tikhonov (z)
    y[4*ne: 5*ne] = eps[2] * laplacian_op_for(x, nx, nz) # 2nd order Tikhonov
    #print(f'total_op_for: y shape: {y.shape}, x shape: {x.shape}')
    return y

#------------------------------------------------total_op_adj
def total_op_adj(y):
    #print(y.shape, mGrad.shape)

    ne = nx * nz
    x = warping_op_adj(y[0: ne]) # Warp term
    #print(f'>>>>>>>>>>>>>>>>>>>>> total_op_adj {x.shape}, {y[ne:2*ne].shape}')
    #import sys
    #sys.exit()
    x = x + eps[0] * y[ne:2*ne] # 0th order Tikhonov
    x = x + eps[1] * derivative1_op_x_adj(y[2*ne : 3*ne], nx, nz) # 1st order Tikhonov (x)
    x = x + eps[1] * derivative1_op_z_adj(y[3*ne : 4*ne], nx, nz) # 1st order Tikhonov (z)
    x = x + eps[2]*laplacian_op_adj(y[4*ne: 5*ne], nx, nz) # 2nd order Tikhonov

    #print(f'total_op_adj: y shape: {y.shape}, x shape: {x.shape}')
    return x

#----------------------------------------------warping_op_for
def warping_op_for(x):
    
    #y = np.reshape(mGrad, (nz, nx)) * np.reshape(x, (nz,nx))
    y = np.reshape(mGrad, (nx, nz)) * np.reshape(x, (nx, nz))
    y = y.reshape((nx*nz))
    #print(f'warping_op_for: --------------- y.shape = {y.flatten().shape}')
    return y

#----------------------------------------------warping_op_adj
def warping_op_adj(y):
    #print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>warping_op_adj: {y.shape}, {mGrad.shape}')
    x = mGrad * y
    return x.flatten()

#-----------------------------------------------derivative1_op_x_for
def derivative1_op_x_for(x, nx, nz):
    '''
    x = np.reshape(x, (nz, nx))
    y = np.zeros(x.shape)
    # check this if the matlab and python codes are equivalent
    #y = [x(:,2:end)-x(:,1:end-1) zeros(nz,1)]; 
    ##y = np.concatenate((x[:,1:] - x[:,0:-1], np.zeros((nz , 1))), axis=1)
    y[:, 0:-1] = np.diff(x)

    y = np.reshape(y, nz*nx)
    '''   
    #print(f'nz = {nz} ,nx= {nx}')
    ne = nx*nz
    x  = np.reshape(x, (nx, nz))
    y  = np.concatenate((x[1:nx,:]-x[0:nx-1,:], np.zeros((1,nz))), axis=0)
    # method 2
    #y  = np.concatenate((np.diff(x, axis=0), np.zeros((1,nz))), axis=0)
    # method 3
    #y = np.zeros(x.shape)
    #y[0:-1, :] = np.diff(x, axis=0)

    y  = np.reshape(y, ne)
    return y

#--------------------------------------------------derivative1_op_x_adj
def derivative1_op_x_adj(y, nx, nz):
    '''
    y = np.reshape(y, (nz, nx))
    #x = [- y(:,1   ) y(:,1:end-2)-y(:,2:end-1) y(:,nx-1)];
    x = np.column_stack((-y[:,0], y[:,0:-2]-y[:,1:-1], y[:,nx-1]))
    #x = np.concatenate((-y[:,0], y[:,0:-2]-y[:,1:-1], y[:,-1]), axis=1)
    # 10/01/23
    #x = np.zeros(y.shape)
    #x[0, :] = -y[0, :]
    #x[1:-1, :] = np.diff(y[1:,:], axis=0)
    #x[-1,:] = y[-1, :]
    x = np.reshape(x, nz*nx)
    '''
    ne = nx*nz
    y  = np.reshape(y, (nx, nz))
    x  = np.concatenate((-y[0:1,:], y[0:nx-2,:]-y[1:nx-1,:], y[nx-2:nx-1,:]), axis=0)
    x  = np.reshape(x, ne)

    return x
        
#------------------------------------------------------derivative1_op_z_for
def derivative1_op_z_for(x, nx, nz):
    '''
    #print(nz, nx)
    x = x.reshape((nz, nx))
    y = np.zeros(x.shape)
    # y = [x(2:end,:)-x(1:end-1,:); zeros(1,nx)];
    ##y = np.concatenate((x[1:,:]-x[0:-1,:], np.zeros((1, nx))), axis=0)
    y[:,0:-1] = np.diff(x, axis=1)
    #print(y.shape)
    y = y.reshape((nz*nx))
    '''
    ne = nx*nz
    x  = np.reshape(x, (nx, nz))
    y  = np.concatenate((x[:,1:nz]-x[:,0:nz-1], np.zeros((nx,1))), axis=1)
    y  = np.reshape(y, ne)

    return y

#--------------------------------------------------derivative1_op_z_adj
def derivative1_op_z_adj(y, nx, nz):
    '''
    y = np.reshape(y,(nz, nx))
    #x = [- y(1,:  ); y(1:end-2,:)-y(2:end-1,:); y(nz-1,:)];
    #x = np.vstack((-y[0:1,:], y[0:-2,:]- y[1:-1,:], y[nz-1:nz,:]))  #, axis=0)
    # 10/01/23
    x = np.zeros(y.shape)
    x[:,0] = -y[:,0]
    x[:,1:-1] = np.diff(y[:,1:], axis=1)
    x[:,-1] = y[:,-1]
    #/10/01/23
    x = np.reshape(x, nz*nx)
    '''
    ne = nx*nz
    y  = np.reshape(y, (nx, nz))
    x  = np.concatenate((-y[:,0:1], y[:,0:nz-2]-y[:,1:nz-1], y[:,nz-2:nz-1]), axis=1)
    x  = np.reshape(x, ne)

    return x

#---------------------------------------------------derivative2_op_x_for
def derivative2_op_x_for(x, nx, nz):
    t =  derivative1_op_x_for(x, nx, nz)
    y =  derivative1_op_x_adj(t, nx, nz)
    #print(f'{y.shape}')

    return y

#-----------------------------------------------derivative2_op_x_adj
def derivative2_op_x_adj(y, nx, nz):
    t =  derivative1_op_x_for(y, nx, nz)
    x =  derivative1_op_x_adj(t, nx, nz)
    return x
#--------------------------------------------------derivative2_op_z_for
def derivative2_op_z_for(x, nx, nz):
    t =  derivative1_op_z_for(x, nx, nz)
    y =  derivative1_op_z_adj(t, nx, nz)
    #print(f'{y.shape}, {t.shape}')
    return y

#---------------------------------------------------derivative2_op_z_adj
def derivative2_op_z_adj(y, nx, nz):
    t =  derivative1_op_z_for(y, nx, nz)
    x =  derivative1_op_z_adj(t, nx, nz)

    return x
#--------------------------------------------------------laplacian_op_for
def laplacian_op_for(x, nx, nz):
    t1 = derivative2_op_x_for(x, nx, nz)
    t2 = derivative2_op_z_for(x, nx, nz)
    #print(f't1 shape = {t1.shape}, t shape = {t2.shape}')
    y = t1 + t2

    return y

#-------------------------------------------------------laplacian_op_adj
def laplacian_op_adj(y, nx, nz):

    t1 = derivative2_op_x_adj(y, nx, nz)
    t2 = derivative2_op_z_adj(y, nx, nz)
    x = t1 + t2

    return x

#---------------------------------------find_warp
# To run with run_idwt.py use  this definition
def find_warp (imageb, imagem, inpara):
# for testing
#def find_warp (imageb, imagem, eps, nlsd_itermax, lsqr_itermax):
    
    globals()['nx'] = imageb.shape[0]
    globals()['nz'] = imageb.shape[1]
    try:
        taper = taper_func(nx, nz)
    except:
        taper = taper_func(nz, nx)

    imagem[:, 0: len(taper)] = imagem[:,0: len(taper)] * taper[:]
    imageb[:, 0: len(taper)] = imageb[:, 0: len(taper)] * taper[:]

    #globals()['eps'] = eps
    globals()['eps'] = [inpara.eps0, inpara.eps1, inpara.eps2]
    nlsd_itermax = inpara.nlsd_itermax
    lsqr_itermax = inpara.lsqr_itermax

    # step 1
    shifts = np.zeros(imagem.shape)
    #print(f'shift shape = {shifts.shape}, imagem shape = {imagem.shape}')

    for ii in range(nlsd_itermax):
        mShift = sincinterp(imagem, shifts, 20)
        globals()['mGrad'] = derivative1_op_z_for(mShift.flatten(), nx, nz)

        r1 = np.reshape(imageb.flatten()- mShift.flatten(), (nx*nz))
        r2 = np.reshape(-shifts, (nz*nx))
        r3 = -derivative1_op_x_for(shifts.flatten(), nx, nz)
        r4 = -derivative1_op_z_for(shifts.flatten(), nx, nz)
        r5 = -laplacian_op_for(shifts.flatten(), nx, nz)
        #print(r1.shape,  r2.shape, r3.shape, r4.shape, r5.shape)

        rhs = np.concatenate((r1, eps[0]*r2, eps[1]*r3, eps[1]*r4, eps[2]*r5))
        #print(f'----------------------rhs shape : {rhs.shape}')
        # python equivalent of matlab lsqr
	#A = LinearOperator((m,n), matvec=Ax, rmatvec=Atb)
        #result = scipy.sparse.linalg.lsqr(A, b)
        
        ne = nx * nz
        L = LinearOperator((5*ne, ne), matvec=total_op_for, rmatvec=total_op_adj)
        sol = lsqr(L, rhs, damp=1, iter_lim=lsqr_itermax, show=False) #, atol=1.0e-5, btol=1.0e-5)
        print(f'Warping inversion iter= {ii}, lsqr istop = {sol[1]}') #, lsqr_maxiteration = {sol[2]}')
        #print(f'norm(r) = {sol[3]}, sol[0] shape = {sol[0].shape}')   

        shifts = shifts + np.reshape(sol[0], (nx, nz))
        #print(f'shift shape = {shifts.shape}')
    #Apply shift
    mShift = sincinterp(imagem, shifts, 20)

    return shifts, mShift

#--------------------------------------------------------sincinterp
def sincinterp(x, ts, n):

    ts = -ts
    #nz, nx = x.shape
    # 12/01/23
    nx, nz = x.shape
    #print(x.shape, nx,nz)
    #print(f'shift shape = {ts.shape}, imagem.shape = {x.shape}')
    Nsinc = 1 + 2 * n
    win = np.empty((1, Nsinc))
    win[:,:] = np.linspace(0, 2*n, 2*n +1) 


    pyround = np.vectorize(lambda x: round(x))
    #i = reshape(bsxfun(@plus,repmat((1:nz),1,nx)+round(-ts(:))',repmat(win',1,nz*nx)),Nsinc,nz,nx);
    term1 = np.tile(np.linspace(0, nz-1, nz), (1, nx)) + np.transpose(pyround(-1.0 * ts.flatten())) 
    # 13/01/2023
    #term1 = np.tile(np.linspace(0, nx-1, nx), (1, nz)) + np.transpose(pyround(-1.0 * ts.flatten())) 

    term2 = np.tile(np.transpose(win), (1, nz*nx))
    i_arr = np.empty(term2.shape)
    for i in range(Nsinc):
        i_arr[i,:] = term1[:,:] + term2[i,:]
    
    #mask = i>nz|i<1;
    mask = (i_arr > nz-1) | (i_arr < 0)
    # 13/01/2023
    #mask = (i_arr > nx-1) | (i_arr < 0)

    #i = reshape(bsxfun(@plus,(1:nz*nx)+round(-ts(:))',repmat(win',1,nz*nx)),Nsinc,nz,nx);
    term1 = np.linspace(0, nz*nx -1, nz*nx, dtype='int') + np.transpose(pyround(-1. * ts.flatten()))
    term2 = np.tile(np.transpose(win) , (1, nz*nx)) 
    i_arr = np.empty(term2.shape)
    for i in range(Nsinc):
        i_arr[i,:] = term1 + term2[i,:]
    i_masked = i_arr[mask == False]

    #j = reshape(repmat(1:nx*nz,Nsinc,1),Nsinc,nz,nx);
    j_arr = np.tile(np.linspace(0, nx*nz -1,  nx*nz, dtype='int'), (Nsinc, 1))
    j_masked = j_arr[mask == False]

    #val = sinc(permute(bsxfun(@plus,reshape(win,[1,1,Nsinc]),ts-round(ts)),[3,1,2]));
    term1 = np.ones((x.shape[0], x.shape[1], Nsinc))
    for i in range(Nsinc):
        term1[:,:,i] = win[0,i] * term1[:,:,i]
    term2 =  ts - pyround(ts)
    for i in range(Nsinc):
        term1[:,:,i] = term1[:,:, i] + term2[:,:]
    val = sinc(np.transpose(term1, (2, 0, 1)))
    val = np.reshape(val, (Nsinc,x.shape[0]*x.shape[1]))
    val_masked = val[mask == False]

    #y = reshape(x(:)'*sparse(i(~mask),j(~mask),val(~mask),nz*nx,nz*nx),nz,nx);
    term1 = np.empty((1, nx*nz))
    term1[:,:] = np.transpose(x.flatten())
    for iii in i_masked:
        if iii < 0:
            print(f'warping function: [sincinterp]: negative index found')
            sys.exit()
    term2 =    csr_matrix((val_masked, (i_masked.astype(int), \
                           j_masked.astype(int))), shape=(nz*nx, nz*nx))

    
    y = np.reshape( term1 * term2, (nx, nz))

    return y
#------------------------------------------sinc
def sinc(x):

    idx = np.where(x == 0)                                                              
    x[idx] = 1.    
       
    y = np.divide(np.sin(np.pi * x), (np.pi * x))                                   
    y[idx] = 1.

    return y

#--------------------------------------- test function
def test_function():
    '''
    To test different functions used in find_shift.
    Never run test_function with the find shift for the real problem,
    since test function changes value of nx and nz globally.
    '''
    nz = 10
    nx = 7
    ne = nz*nx

    globals()['nx'] = nx ; globals()['nz'] = nz
    x = rand(ne); y = rand(ne)
    yy = derivative1_op_x_for(x, nx, nz)
    xx = derivative1_op_x_adj(y, nx, nz)
    print(f'derivative1_op_x')
    print(f'x . xx = {np.inner(x, xx)}, y . yy = {np.inner(y,yy)}')

    x = rand(ne); y = rand(ne)
    yy = derivative1_op_z_for(x, nx, nz)
    xx = derivative1_op_z_adj(y, nx, nz)
    print(f'derivative1_op_z')
    print(f'x . xx = {np.inner(x, xx)}, y . yy = {np.inner(y,yy)}')
   
    x = rand(ne); y = rand(ne)
    yy = derivative2_op_x_for(x, nx, nz)
    xx = derivative2_op_x_adj(y, nx, nz)
    print(f'derivative2_op_c')
    print(f'x . xx = {np.inner(x, xx)}, y . yy = {np.inner(y,yy)}')

    x = rand(ne); y = rand(ne)
    yy = derivative2_op_z_for(x, nx, nz)
    xx = derivative2_op_z_adj(y, nx, nz)
    print(f'derivative2_op_z')
    print(f'x . xx = {np.inner(x, xx)}, y . yy = {np.inner(y,yy)}')   

    x = rand(ne); y = rand(ne)
    yy = laplacian_op_for(x, nx, nz)
    xx = laplacian_op_adj(y, nx, nz)
    print(f'laplacian_op')
    print(f'x . xx = {np.inner(x, xx)}, y . yy = {np.inner(y,yy)}')

    globals()['x'] = rand(ne); globals()['y'] = rand(ne); globals()['mGrad'] =rand(ne)
    yy = warping_op_for(x)
    xx = warping_op_adj(y)
    quit()
    x = rand(ne); y = rand(5*ne); globals()['mGrad'] = rand(ne)
    globals()['eps'] = [1., 1., 1.]
    globals()['nx'] = nx
    globals()['nz'] = nz
    yy = total_op_for(x)
    xx = total_op_adj(y)
    print(f'total_op')
    print(f'x . xx = {np.inner(x, xx)}, y . yy = {np.inner(y,yy)}')
    quit()

    # works
    y = rand(5*ne); globals()['mGrad'] = rand(ne)
    globals()['eps'] = [1., 1., 1.]
    L = LinearOperator((5*ne, ne), matvec=total_op_for, rmatvec=total_op_adj)

#---------------------------------- taper
def taper_func(nx, nz):
    nzeros = 20
    ntaper = 100
    nones = 121

    # taper=[zeros(nzeros,1); 0.5*(1-cos(([1:ntaper].'-1)/ntaper*pi)); ones(nones,1)]*ones(1,nx);
    taper = np.concatenate((np.zeros(nzeros),
                           0.5 * (1 - np.cos((np.linspace(0,ntaper-1).T - 1.) / ntaper * np.pi)),
                           np.ones(nones)))
    taper = taper[:] * np.ones(nx)[0:taper.shape[0]]
    
    return taper
