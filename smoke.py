import numpy as np
from utils import sample_points_on_disc

def smoke(current_time, prev_time, particles, params):
    '''
    :param: current_time: thecurrenttime[sec](double1-by-1) 2) 
    :param: prev_time: the previous time in[sec](double1-by-1) 
    :param: particles: struct (dict? DF?) of dimensions [N, 3 (x,y,z), 3](x,v,a):
        a. X : position [m], velocity [m/sec] and acceleration [m/sec^2] in the x-axis (double 1-by-3)
        b. Y : same as “X” for the y-axis
        c. Z : same as “X” for the z-axis
     :param: params: struct with the fields:
        a. particles_per_sec:  number of particles created every second (double 1-by- )1
        b. ini_vel: theinitialvelocityoftheparticles[m/sec](double1-by-3)
        c. vel_std: the standard deviation of the initial velocity [m/sec] (double 1-by-3)
        d. bouyancy: the buoyancy strength[m/sec^2](double1-by-1)
        e. buoyancy_const: the buoyancy time constant [sec] (double 1-by-1)
        f. wind_vel: the planar wind velocity [m/sec] (double 1-by-2)
        g. drag: the drag coefficient(double1-by-1)
        h. aperture: the diameter of the chimney [m] (double 1-by-1)

    solution:
    the sum of acceleration for particle i follows the formula:
        a_i = -g + a_wind + a_bouyancy = -g + D(v_wind - v_i)^2 
    '''
    
    # constants:
    g = 9.81 # m/s^2
    air_viscosity = 1.81e-5 # kg/(m*sec)
    particle_radius = 10**-6 # meters
    particle_mass = 10**-15 # kg (calculated for water droplet)

    alpha = 6*np.pi*air_viscosity*particle_radius/particle_mass
    t = current_time - prev_time
    
    # NEW PARTICLES array of new particles: sample initial positions, t_c, and velocity, calculate the acceleration.
    num_particles_to_create = int(params['particles_per_sec']*t)
    aperture_radius = params['aperture']/2


    ceation_times = np.random.uniform(0, t, num_particles_to_create)

    creation_positions_xy = sample_points_on_disc(aperture_radius, num_particles_to_create)
    creation_positions = np.concatenate((creation_positions_xy, np.zeros((num_particles_to_create, 1))), axis=1) #[N, 3]
    
    creation_velocities = np.random.normal(params['ini_vel'], params['vel_std'], (num_particles_to_create, 3)) #[N, 3]

    creation_bouyancies = np.ones((num_particles_to_create, 1))*params['bouyancy']

    # todo: we want to do the same formula for both, accept for the top there are different creation times and the same bouyancy and in the bottom different bouyancies and creation time 0
    #GIVEN PARTICLES: for them you need to calculate the initial bouyancy    
    existing_positions = particles[:,:,0]
    existing_velocities = particles[:,:,1]
    existing_accelerations = particles[:,:,2]

    existing_bouyancies = existing_accelerations[:,2] + g + alpha*existing_velocities[:,2]
    existing_bouyancies = np.expand_dims(existing_bouyancies, axis=1)

    
    

    particles_updated = particles
    return particles_updated