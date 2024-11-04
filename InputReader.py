import configparser
import sys
#-----------------------------------------
class Inparam:
    def __init__(self, inp_idwt_file):
        print('InpReader.py', '[INPUT]: Creating input class')
        config_ini = configparser.ConfigParser()
        config_ini.read(inp_idwt_file)
        # ---------------section_model
        self.model_path = eval(config_ini.get('section_model', 'model_path'))
        #self.origin_model_shape = eval(config_ini.get('section_model', 'origin_model_shape'))
        #print(self.origin_model_shape[0])
        #self.scale_down = float(config_ini.get('section_model', 'scale_down'))
        self.shape = eval(config_ini.get('section_model', 'shape'))
        #self.extent = eval(config_ini.get('section_model', 'extent'))
        self.spacing = eval(config_ini.get('section_model', 'spacing'))
        self.origin = eval(config_ini.get('section_model', 'origin'))
        self.bcs = eval(config_ini.get('section_model', 'bcs'))
        self.nbl = int(config_ini.get('section_model', 'nbl'))

        self.CFL = eval(config_ini.get('section_model', 'CFL'))
     
        #self.smoothing_shape = eval(config_ini.get('section_smooth', 'smoothing_shape'))
        #self.plume_sigma = float(config_ini.get('section_smooth', 'plume_sigma'))
        self.sigma = eval(config_ini.get('section_smooth', 'sigma'))

        self.gauss_amp = float(config_ini.get('section_monitor', 'gauss_amp'))
        self.gauss_width = float(config_ini.get('section_monitor', 'gauss_width'))


        self.nshots = int(config_ini.get('section_src', 'nshots'))
        self.src_depth = float(config_ini.get('section_src', 'src_depth'))
        self.src_type = eval(config_ini.get('section_src', 'src_type'))
        self.f0 = float(config_ini.get('section_src', 'f0'))

        self.nrec = int(config_ini.get('section_receiver', 'nrec'))
        self.rec_depth = float(config_ini.get('section_receiver', 'rec_depth'))
        self.mask_rec_data = eval(config_ini.get('section_receiver', 'mask_rec_data'))

        self.t0 = float(config_ini.get('section_simulation', 't0'))
        self.tn = float(config_ini.get('section_simulation', 'tn'))
        self.time_order = int(config_ini.get('section_simulation', 'time_order'))
        self.space_order = int(config_ini.get('section_simulation', 'space_order'))
        self.forward_space_order = int(config_ini.get('section_simulation', 'forward_space_order'))

        self.outpath = eval(config_ini.get('section_output', 'outpath'))
        
        self.eps0 = eval(config_ini.get('section_warp', 'zero_order_regularisation_param'))
        self.eps1 = eval(config_ini.get('section_warp', 'first_order_regularisation_param'))
        self.eps2 = eval(config_ini.get('section_warp', 'smoothing_param'))
        self.nlsd_itermax = int(config_ini.get('section_warp', 'inversion_iter'))
        self.lsqr_itermax = int(config_ini.get('section_warp', 'lsqr_itermax'))


        self.water_level = int(config_ini.get('section_alfa', 'water_level'))
        self.ntaper = int(config_ini.get('section_alfa', 'ntaper'))

        self.grad_type = eval(config_ini.get('section_gradient', 'grad_type'))
        self.grad_smooth = int(config_ini.get('section_gradient', 'grad_smooth'))
        xcord = eval(config_ini.get('section_gradient', 'mask_xmin_xmax'))
        zcord = eval(config_ini.get('section_gradient', 'mask_zmin_zmax'))

        self.mask_xmin = xcord[0]
        self.mask_xmax = xcord[1]
        self.mask_zmin = zcord[0]
        self.mask_zmax = zcord[1]


        self.n_iter = int(config_ini.get('section_inversion', 'n_iter'))
        self.learning_rate = float(config_ini.get('section_inversion', 'learning_rate'))
        self.inv_tol = float(config_ini.get('section_inversion', 'stop_tolerance'))

