import numpy as np
import warnings
from utils import sample_points_on_disc

def smoke(current_time, prev_time, particles, params):
    '''
    :param: current_time: thecurrenttime[sec](double1-by-1) 2) 
    :param: prev_time: the previous time in[sec](double1-by-1) 
    :param: particles: np array of dimensions [N, 3 (pos,vel,acc), 3(x,y,z)]:
        
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
    assert current_time > prev_time, "current_time <= prev_time"

    g = 9.81 # m/s^2
    
    drag = params['drag']
    t = current_time - prev_time
    tau = params['buoyancy_const']
    wind_vel = params['wind_vel']
    assert drag != tau, "alpha = tau => point discontinutity, not implemented yet."  
    
    # NEW PARTICLES array of new particles: sample initial positions, t_c, and velocity, calculate the acceleration.
    num_particles_to_create = round(params['particles_per_sec']*t) 
    if num_particles_to_create == 0:
        warnings.warn("number of particles to create is 0, no new particles will be created.")
    
    aperture_radius = params['aperture']/2

    creation_times = np.random.uniform(0, t, num_particles_to_create) #[N,1]

    creation_positions_xy = sample_points_on_disc(aperture_radius, num_particles_to_create)
    creation_positions = np.concatenate((creation_positions_xy, np.zeros((num_particles_to_create, 1))), axis=1) #[N, 3]
    
    creation_velocities = np.random.normal(params['ini_vel'], params['vel_std'], (num_particles_to_create, 3)) #[N, 3]

    creation_buoyancies = np.ones(num_particles_to_create)*params['bouyancy']

    #GIVEN PARTICLES: for them you need to calculate the initial bouyancy    
    # if particles.size == 0:
    #     particles = np.zeros((0,3,3))
    existing_positions = particles[:, 0, :] #[N,3]
    existing_velocities = particles[:, 1, :] #[N,3]
    existing_accelerations = particles[:, 2, :] #[N,3]

    existing_buoyancies = existing_accelerations[:, 2] + g + drag*existing_velocities[:, 2]
    # existing_bouyancies = existing_bouyancies #[N,1]

    existing_times = np.zeros_like(existing_buoyancies) #[N,1]

    # concatenate creation conditions and creation times
    tc_list = np.concatenate((creation_times, existing_times), axis=0) #[2N,1]

    all_positions_tc = np.concatenate((creation_positions, existing_positions), axis=0) #[2N,3]
    all_velocities_tc = np.concatenate((creation_velocities, existing_velocities), axis=0) #[2N,3]
    all_buoyancies_tc = np.concatenate((creation_buoyancies, existing_buoyancies), axis=0) #[2N,1]
    all_accelerations_tc = -g + np.expand_dims(all_buoyancies_tc, axis=1) + drag*all_velocities_tc
    
    # calculate kinematics
    
    init_z = all_positions_tc[:, 2]
    init_vz = all_velocities_tc[:, 2]

    init_x = all_positions_tc[:, 0]
    init_vx = all_velocities_tc[:, 0]
    
    init_y = all_positions_tc[:, 1]
    init_vy = all_velocities_tc[:, 1]

    all_sigma = tau*all_buoyancies_tc/(1-drag*tau)

    # z
    z_t = init_z - tau*all_sigma*(np.exp(-(t-tc_list)/tau) - 1) \
        + 1/drag*(init_vz + g/drag + all_sigma)*(1 - np.exp(-drag*(t-tc_list)))\
        - g/drag*(t-tc_list)

    vz_t = init_vz*np.exp(-drag*(t-tc_list)) \
          + all_sigma*(np.exp(-drag*(t-tc_list)) - np.exp(-(t-tc_list)/tau)) \
          + g/drag*(np.exp(-drag*(t-tc_list)) - 1)
    
    az_t = -drag*init_vz*np.exp(-drag*(t-tc_list)) \
          + all_sigma*(-drag*np.exp(-drag*(t-tc_list)) + 1/tau*np.exp(-(t-tc_list)/tau)) \
          - g*drag*np.exp(-drag*(t-tc_list))
    
    # x
    x_t = init_x + 1/drag*(init_vx - wind_vel[0])*(1 - np.exp(-drag*(t-tc_list))) \
        + wind_vel[0]*(t-tc_list)
    
    vx_t = wind_vel[0] + (init_vx - wind_vel[0])*np.exp(-drag*(t-tc_list))
    ax_t = -drag*(init_vx - wind_vel[0])*np.exp(-drag*(t-tc_list)) 

    # y exactly the same as x 
    y_t = init_y + 1/drag*(init_vy - wind_vel[1])*(1 - np.exp(-drag*(t-tc_list))) \
        + wind_vel[1]*(t-tc_list)
    
    vy_t = wind_vel[1] + (init_vy - wind_vel[1])*np.exp(-drag*(t-tc_list))
    ay_t = -drag*(init_vy - wind_vel[1])*np.exp(-drag*(t-tc_list))

    # concatenate the results into a tensor of dimensions [N, 3(pos,vel,acc), 3(x,y,z)]
    positions_updated = np.stack((x_t, y_t, z_t), axis=1) #[N,3]
    velocities_updated = np.stack((vx_t, vy_t, vz_t), axis=1) #[N,3]
    accelerations_updated = np.stack((ax_t, ay_t, az_t), axis=1) #[N,3]

    particles_updated = np.stack((positions_updated, velocities_updated, accelerations_updated), axis=1) #[N,3,3]

    buoyancies_updated = all_buoyancies_tc*np.exp(-(t-tc_list)/tau)
    return particles_updated