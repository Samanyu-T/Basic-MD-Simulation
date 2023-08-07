''' The main class requires as input a set of co-ordinates of the atom locations in the crystal'''
class Molecular_Dynamics:
    
    def __init__(self,xyz, box, vel = None, cutoff = 6, mr = 183.84):
        
        
        ''' To Initialize the Parameters such as Potential, Force and Hessian use Update Properties '''
        
        self.spline_phi = np.array([
            
            [
                 0.954071477542914e2,
                -0.181161004448916e3,
                 0.930215233132627e2,
                -0.108428907515182e2,
                 0.112027468539573e2,
                -0.312459176640227e1,
                 0.123028140617302e1,
                 0.154767467307454e1,
                -0.128861387780439e1,
                -0.843327493551467e0,
                 0.214009882965042e1,
                -0.102898314532388e1,
                 0.138163259601912e1,
                -0.360872433001551e1,
                 0.217655968740690e1
                
            ],
            
            [
                2.564897500000000,
                2.629795000000000,
                2.694692500000000,
                2.866317500000000,
                2.973045000000000,
                3.079772500000000,
                3.516472500000000,
                3.846445000000000,
                4.176417500000000,
                4.700845000000000,
                4.895300000000000,
                5.089755000000000,
                5.342952500000000,
                5.401695000000000,
                5.460437500000000
            ]
        ])
        
        self.spline_rho = np.array([
            
            [ 
                -0.420429107805055e1,
                 0.518217702261442e0,
                 0.562720834534370e-1,
                 0.344164178842340e-1
                
            ],

            [
                2.500000000000000,
                3.100000000000000,
                3.500000000000000,
                4.900000000000000
            ]
        ])
        
        self.spline_F = np.array([-5.553986589859130,
                                  -0.045691157657292])
        
        #Initialize all the attributes
    
        self.potential = None
        self.force = None
        self.hessian = None
        
        self.vel = vel
        self.box = box
        self.xyz = xyz
        self.cutoff = cutoff

        # Convert 1 amu to eV(ps/A)^2
        amu_conv = 931.49432e6*2.99792458e6**-2

        #Find the mass of a single atom in eV(ps/A)^2
        self.m = mr*amu_conv

        #Apply Periodic Boundary Conditions to the position
        self.Apply_PBC_xyz()
        
    def Apply_PBC_xyz(self):
        
        #Round the values for more numerical stability - prevent truncation errors
        self.xyz = np.round(self.xyz, decimals = 10)

        #Standard Equation for PBC - wiki
        for idx, pos in enumerate(self.xyz):
            self.xyz[idx] = pos - np.floor(pos / self.box[0]) * self.box[0]
        
        #Evaluate the KDTree and find the neighbours to each atom
        kdtree = KDTree(self.xyz, boxsize = self.box)
        self.neighbors = kdtree.query_ball_point(self.xyz , self.cutoff, np.inf)
    
    def Update_Properties(self, calc_potential = True, calc_force = True, calc_hessian = True):
        
        ''' Simply Updates the Properties: Potential, Force and Hessian, if a property does not need to be update simply input False'''
        if calc_potential:
            self.potential = self.Potential()
        if calc_force:
            self.force = self.Force()
        if calc_hessian:
            self.hessian = self.Hessian()
        
    def init_velocity(self, T = 300):

        '''Initialize the Velocities of each Atom based on the Maxwel Boltzmann Distribution'''
        k = 8.617333262e-5
        
        std = np.sqrt(k*T/(self.m))

        self.vel = np.hstack([
                
                np.random.normal(loc = 0, scale = std, size = (len(self.xyz),1)),
                np.random.normal(loc = 0, scale = std, size = (len(self.xyz),1)),
                np.random.normal(loc = 0, scale = std, size = (len(self.xyz),1))

        ])
 
    def Phi(self, dij):
        ''' Evaluate the Interatomic Pairwise Potential - using Spline Functions'''

        mask_phi = (self.spline_phi[1, :] > dij).astype(int)
        phi = np.sum(mask_phi * self.spline_phi[0, :] * 
                         (self.spline_phi[1, :] - dij)**3, axis=1)
        
        # Divide by 2 to account for double counting of the potentials
        return phi/2
        
    def Density(self, dij):
        ''' Evaluate the Electron Density Surrounding an Atom - using Spline Functions'''

        dij = np.clip(dij, a_min = 2.002970124727, a_max = None)
        mask_rho = (self.spline_rho[1, :] > dij).astype(int)
        density = np.sum(mask_rho * self.spline_rho[0, :] * 
                 (self.spline_rho[1, :] - dij)**3, axis=1)
        return density
    
    def F(self, density):
        ''' Evaluate the resulting potential from the Electron Density around an atom'''

        F = self.spline_F[0] * np.sqrt(density) + self.spline_F[1] * density **2
        return F
    
    def Distance_Enum(self, i, j_all):
        ''' Applying the Periodic Boundary Conditions using Enumeration and seeing if wrapping the atom around optimizes the distance'''

        a = [0, 1, -1]
        combinations = list(itertools.product(a, repeat=3))

        rij = np.zeros([len(j_all),3])
        dij = np.zeros([len(j_all),1])
        
        for l,j in enumerate(j_all):

            enum = np.array(
                 [ np.linalg.norm(self.xyz[i] - self.xyz[j] - k*self.box)  for k in combinations] 
                           )
          
            idx  = np.argmin(enum)
            dij[l] = enum[idx]
            rij[l] = self.xyz[i] - self.xyz[j] - combinations[idx]

        return rij, dij
    
    def Distance(self, xyz_i, xyz_j, delta = np.zeros(3)):
        ''' Applying the Periodic Boundary Conditions using Standard Equation from wiki'''

        xyz_ij = xyz_i + delta - xyz_j

        for idx, r in enumerate(xyz_ij):
            for idx_d, dim in enumerate(r):
                
                if dim < -self.box[0]/2 :
                    xyz_ij[idx][idx_d] = dim + self.box[0]

                if dim > self.box[0]/2 :
                    xyz_ij[idx][idx_d] = dim - self.box[0]
                    
        return xyz_ij, np.linalg.norm(xyz_ij, axis=1)[:,np.newaxis]
                     
    def Potential(self):
        ''' Evaluate the Potential for every atom in the Crystal'''

        #Initialize
        potential = np.zeros(len(self.xyz))
        
        #Loop through each atom in the crystal
        for i, j in enumerate(self.neighbors):
            
            #Remove the atom of interest from the list of neighbours
            j = [k for k in j if k != i]
            
            rij, dij = self.Distance(self.xyz[i],self.xyz[j], np.zeros((3,)))

            #Find each contribution to the potential - sum each neighbor for total contribution
            phi = np.sum(self.Phi(dij),axis = 0)
            rho = np.sum(self.Density(dij),axis = 0)
            F = self.F(rho)
            
            #The total potential is the sum of contributions
            potential[i] = phi + F
            
        return potential
    
    def dPhi(self, dij):
        ''' The First Derivative of the Spline Function for interatomic pairwise potentials. (Phi)'''
        mask_phi = (self.spline_phi[1, :] > dij).astype(int)
        dphi = -3*np.sum(mask_phi * self.spline_phi[0, :] * 
                         (self.spline_phi[1, :] - dij)**2, axis=1)
        return dphi/2
    
    def dRho(self, dij):
        ''' The First Derivative of the Spline Function for electron density. (Rho)'''

        a_min = 2.002970124727
        
        mask_rho = (self.spline_rho[1, :] > dij).astype(int)
        mask_dij = (dij > a_min ).astype(int)
        drho = -3*np.sum(mask_rho * self.spline_rho[0, :] * 
                 (self.spline_rho[1, :] - dij)**2, axis=1) * mask_dij[:,0]
        return drho
    
    def dF(self, density):
        ''' The First Derivative of F (potential resulting from electron density) '''

        dF = 0.5*self.spline_F[0] * density**(-0.5) + 2*self.spline_F[1] * density
        return dF
    
    def Force(self):
        ''' Evaluate the force on each atom given a Crystal Structure'''

        force = np.zeros([len(self.xyz),3])
        
        #Loop through each atom in the crystal
        for i, j in enumerate(self.neighbors):
            
            #Remove the atom of interest from the list of neighbours
            j = [k for k in j if k != i]
            
            rij, dij = self.Distance(self.xyz[i],self.xyz[j], np.zeros((3,)))

            # Evaluate the First order derivative of each function
            dphi = self.dPhi(dij)
            
            drho = self.dRho(dij)
            
            rho = np.sum(self.Density(dij),axis = 0)
            
            df = self.dF(rho)
            
            # Combine the derivatives using chain rule Chain Rule
            fij = ((dphi + df*drho)[:,np.newaxis]/dij)*rij
            
            #Sum the forces (fij) to find the resultant force for the ith atom 
            fi = np.sum(fij, axis=0)
            
            force[i] = -fi
            
        return force
    
    def Force_FD(self, delta):
        ''' Calculating the Force using Finite-Differences '''

        #Initialize        
        dx = np.zeros(len(self.xyz))
        dy = np.zeros(len(self.xyz))
        dz = np.zeros(len(self.xyz))
        
        #Evaluate the Potential for a slight pertubation in x
        for i, j in enumerate(self.neighbors):
            j = [k for k in j if k != i]
            
            rij, dij = self.Distance(self.xyz[i],self.xyz[j], delta*np.array([1 , 0 ,0]) )

            phi = np.sum(self.Phi(dij),axis = 0)
            rho = np.sum(self.Density(dij),axis = 0)
            
            
            F = self.F(rho)
            
            dx[i] = phi + F

        #Evaluate the Potential for a slight pertubation in y
        for i, j in enumerate(self.neighbors):
            j = [k for k in j if k != i]
            
            rij, dij = self.Distance(self.xyz[i],self.xyz[j], delta*np.array([0 , 1 ,0]) )

            phi = np.sum(self.Phi(dij),axis = 0)
            rho = np.sum(self.Density(dij),axis = 0)
            
            
            F = self.F(rho)
            
            dy[i] = phi + F
            
        #Evaluate the Potential for a slight pertubation in z
        for i, j in enumerate(self.neighbors):
            j = [k for k in j if k != i]
            
            rij, dij = self.Distance(self.xyz[i],self.xyz[j], delta*np.array([0 , 0 , 1]) )


            phi = np.sum(self.Phi(dij),axis = 0)
            rho = np.sum(self.Density(dij),axis = 0)
            
            
            F = self.F(rho)
            
            dz[i] = phi + F
        
        #Apply the Finite Difference Equation to approximate the force 
        force_approx  = (1/delta)*np.array([dx - self.potential,
                                  dy - self.potential,
                                  dz - self.potential])
        return force_approx.T
    
    def d2Phi(self, dij):
        ''' The Second Derivative of the Spline Function for interatomic interactions. (Phi)'''
        mask_phi = (self.spline_phi[1, :] > dij).astype(int)
        d2phi = 6*np.sum(mask_phi * self.spline_phi[0, :] * 
                         (self.spline_phi[1, :] - dij), axis=1)
        return d2phi/2
    
    def d2Rho(self, dij):
        ''' The Second Derivative of the Spline Function for electron density. (Rho)'''

        a_min = 2.002970124727
        mask_dij = (dij > a_min ).astype(int)

        mask_rho = (self.spline_rho[1, :] > dij).astype(int)
        d2rho = 6*np.sum(mask_rho * self.spline_rho[0, :] * 
                         (self.spline_rho[1, :] - dij), axis=1)*mask_dij[:,0]
        return d2rho

    def d2F(self, density):
        ''' The Second Derivative of the Spline Function for the Potential Resulting from the electron density. (F)'''

        d2F = -0.25*self.spline_F[0] * density**(-1.5) + 2*self.spline_F[1]
        return d2F
    
    def Hessian(self):
        ''' Rvaluates the Hessian (wrt xyz) For Each Atom'''
        hessian = np.zeros([len(self.xyz),3, 3])
        
        #Loop through each atom in the crystal
        for i, j in enumerate(self.neighbors):
            
            #Remove the atom of interest from the list of neighbours
            j = [k for k in j if k != i]
            
            rij, dij = self.Distance(self.xyz[i],self.xyz[j], np.array([0 , 0 ,0]) )

            # Evaluate the First order derivative of each function
            dphi = self.dPhi(dij)
            
            drho = self.dRho(dij)
            
            rho = np.sum(self.Density(dij),axis = 0)
            
            df = self.dF(rho)
            
            d2phi = self.d2Phi(dij)
            
            d2rho = self.d2Rho(dij)
            
            d2F = self.d2F(rho)
            
            rij_rijT = np.array([ (x[:,np.newaxis]@x[:,np.newaxis].T)/dij[idx]**2 for idx, x in enumerate(rij)])

            # Combine the derivatives using chain rule Chain Rule
            hij = ((dphi + df*drho)[:,np.newaxis]/dij)[:,:,np.newaxis] * (np.eye(3) - rij_rijT) + \
                  (d2phi + d2rho*df )[:,np.newaxis,np.newaxis] * rij_rijT 
                  
        
            #Sum the forces (fij) to find the resultant force for the ith atom 
            hi = np.sum(hij, axis=0)

            hessian[i] = hi + d2F*np.sum(drho[:,np.newaxis]*rij/dij, axis = 0) @ \
                              np.sum(drho[:,np.newaxis]*rij/dij, axis = 0)
            
        return hessian
    
    def Hessian_FD(self, delta):
        ''' Evaulate Hessian From Finite Differences'''

        force = np.zeros([len(self.xyz),3])
        
        #Initialize        
        dx = np.zeros([len(self.xyz),3])
        
        #Loop through each atom in the crystal
        for i, j in enumerate(self.neighbors):
            
            #Remove the atom of interest from the list of neighbours
            j = [k for k in j if k != i]
            
            rij, dij = self.Distance(self.xyz[i],self.xyz[j], delta*np.array([0 , 1 ,0]) )

            # Evaluate the First order derivative of each function
            dphi = self.dPhi(dij)
            
            drho = self.dRho(dij)
            
            rho = np.sum(self.Density(dij),axis = 0)
            
            df = self.dF(rho)
            
            # Combine the derivatives using chain rule Chain Rule
            fij = ((dphi + df*drho)[:,np.newaxis]/dij)*rij
            
            #Sum the forces (fij) to find the resultant force for the ith atom 
            fi = np.sum(fij, axis=0)
            
            dx[i] = fi
        
        hessian  = (dx - self.force)/delta
        return hessian

    def Optimize_xyz(self, defect_type, alpha = 1, reg = 0, tolerance = 1e-3):
        '''Given an crystal structure optimize that strucute to minimize the potential
           defect_type -> is a string to label the defect
           alpha -> is effectively the step size the maximum is 1 lowering it will slower convergence but the convergence will be smoother
           reg -> is a regularization parameter which can help will smoother convergence and provides numerical stability if the Hessian is ill-conditioned '''

        #Initialize the starting erorr and lists        
        error = 10

        error_lst = [0]
        alpha_lst = []
        iteration = 0
        
        #Update The Properties
        self.Update_Properties()
        
        #Store the current potential
        potential_lst = [sum(self.potential)]
        
        #Loop till termination criterion are met
        while error > tolerance and iteration <= 50:
            
            #Export data to an LAMMPS Dump file for Ovito
            self.export_ovito('ovito_files/' + defect_type + '/',
                        defect_type + '.' + str(iteration) + '.dump')

            #Hessian Based Descent - Newton CG
            for idx, val in enumerate(self.xyz):
                self.xyz[idx] += alpha * np.linalg.solve(self.hessian[idx] + reg*np.eye(3),
                                                        self.force[idx])
            #structure += alpha * Crystal.force
            
            #Apply the PBC and Update the Properties for next iteration
            self.Apply_PBC_xyz()
            self.Update_Properties()

            #Error is defined in this case as the maximum resultant force can be changed to convergance of potential
            error = np.linalg.norm(self.force.max() )
            error_lst.append(error)
            alpha_lst.append(alpha)
            
            potential_lst.append(sum(self.potential))
            
            iteration += 1
            
            #Update of the step size, can be tweaked for better convergence
            if  iteration % 75 == 74:
                alpha = alpha/2
            

        return self.xyz, error_lst[1:], alpha_lst, potential_lst
    
    def Time_Integration(self, dt_max = 1, n_step = 100, max_displacement = 2e-1):
        ''' Apply Velocity Verlet Equation to simulate the dynamics of the crystal'''


        reference_xyz = self.xyz
        reference_neighbors = self.neighbors
        a_disp = []
        #Store the evolution of the total potential
        #total_potential = np.zeros(n_step)
        self.ws_analysis = self.Wigner_Seitz(reference_xyz,reference_neighbors)

        #Loop through the steps
        for i in range(n_step):
            
            #Evaluate the total energy at a given state
            #total_potential[i] = sum(self.potential) + sum(0.5*self.m*np.sum(self.vel**2, axis = 1))

            #Export data to Ovito
            self.export_ovito('ovito_files/' + 'radiation' + '/',
                'radiation' + '.' + str(i) + '.dump')

            # Apply the Velocity Verlet Update
            m_inv = 1/(self.m)

            max_a = m_inv*np.max(np.abs(self.force))

            max_v = np.max(np.abs(self.vel))

            #Update Time Step
            if dt_max*max_v + 0.5*dt_max**2*max_a > max_displacement:

                k = max_v/max_a

                dt = k*(np.sqrt(1+4*k*max_displacement) - 1)
            else:
                dt = dt_max

            self.xyz =  self.xyz + self.vel*dt + 0.5*(self.force*m_inv)*dt**2

            a_disp.append(np.linalg.norm(self.vel*dt + 0.5*(self.force*m_inv)*dt**2).max())

            self.Apply_PBC_xyz()

            vel_temp = self.vel + 0.5*(self.force*m_inv)*dt 

            self.force = self.Force()

            self.vel = vel_temp + 0.5*(self.force*m_inv)*dt

            self.ws_analysis = self.Wigner_Seitz(reference_xyz,reference_neighbors)
            

        #return total_potential
        plt.plot(a_disp)
    

    def Wigner_Seitz(self, reference_xyz, reference_neighbors):
        ws_analysis = np.zeros(len(self.xyz))

        for i, ref_xyz in enumerate(reference_xyz):

            idx_neighbor = reference_neighbors[i]

            _ , d_ws = self.Distance(self.xyz[i], reference_xyz[idx_neighbor])

            min_idx = np.argmin(d_ws)

            ws_analysis[idx_neighbor[min_idx]] += int(1)
        
        return ws_analysis

    def export_ovito(self,folder_path,efile):  
        '''Will Export the necessary data to Ovito, given folder path and file name'''

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        #print ("Exporting data to %s" % efile)
        
        wfile = open(folder_path + '/' + efile, 'w')
        wfile.write("ITEM: TIMESTEP\n")
        wfile.write("0\n")
        wfile.write("ITEM: NUMBER OF ATOMS\n")
        wfile.write("%d\n" % (len(self.xyz)))



        # assume 3D pbc with orthogonal box
        wfile.write("ITEM: BOX BOUNDS pp pp pp\n")
        wfile.write("%f %f\n" % (0.0, self.box[0]))
        wfile.write("%f %f\n" % (0.0, self.box[1]))

        wfile.write("%f %f\n" % (0.0, self.box[2]))

        # we have only one atom species for now

        _type = 1
        if self.vel is None:
            wfile.write("ITEM: ATOMS id type x y z POTENTIAL_ENERGY \n")
            for _i,_xyz in enumerate(self.xyz):

                wfile.write('%3d %3d %14.8f %14.8f %14.8f %14.8f\n' % 
                        (_i, _type, _xyz[0], _xyz[1], _xyz[2], self.potential[_i]))
        else:
            wfile.write("ITEM: ATOMS id type x y z POTENTIAL_ENERGY vx vy vz \n")
            for _i,_xyz in enumerate(self.xyz):

                wfile.write('%3d %3d %14.8f %14.8f %14.8f %14.8f %14.8f %14.8f %14.8f\n' % 
                        (_i, _type, _xyz[0], _xyz[1], _xyz[2], self.potential[_i], self.vel[_i][0] , self.vel[_i][1], self.vel[_i][2]))



        wfile.close()