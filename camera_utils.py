import numpy as np 
import cv2 

from scipy.interpolate import griddata
from scipy import ndimage

def vertex_from_depth(depth,
                      fl,
                      pp,
                      depth_range,
                      png_scale_factor): 
    '''
        This function takes depth map (png image) and 
        converts it into a point cloud 
        
        @params
        fl: focal lenth of the camera

        pp: principal point, center of the camera

        depth_range: depth range from min to max

        pnt_scale_factor: the scale factor by which the 
                          depth maps were scaled to convert
                          to png image 

    '''
        
    fl_x, fl_y = fl 
    pp_x, pp_y = pp 
    min_depth, max_depth = depth_range

    rows, cols = depth.shape

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    # convert depth to values in meters 
    depth_m = np.copy(depth) / png_scale_factor 

    # check for range 
    depth_m[depth_m < min_depth] = min_depth
    depth_m[depth_m > max_depth] = max_depth

    # depth_mm = np.max(depth_mm, 0.0)

    vx = (xp - pp_x) * depth_m / fl_x 
    vy = (yp - pp_y) * depth_m / fl_y 
    vz = depth_m 

    vertices = np.transpose(np.stack([vx, vy, vz]), (1, 2, 0))

    return vertices, depth_m 

def normal_from_vertex(vertices):

    '''
        This function takes a point cloud and 
        returns normal vectors at each point
        
        @params
        vertices: input vertices that are used 
                  to compute the normals via 
                  cross product
    '''

    Vr = np.zeros_like(vertices)
    Vu = np.zeros_like(vertices)

    rows, cols, _ = vertices.shape

    Vu[0: rows-1, :, :] = vertices[1: rows, :, :]
    Vr[:, 0: cols-1, :] = vertices[:, 1:cols,  :]

    a = Vr - vertices
    b = Vu - vertices 

    axb = np.cross(a, b)

    mag = np.linalg.norm(axb, axis=-1)
    mag[mag < 1e-10] = 1 

    axb[...,0] = axb[...,0] / mag 
    axb[...,1] = axb[...,1] / mag 
    axb[...,2] = axb[...,2] / mag 

    return axb 

def add_gaussian_shifts(depth):

    rows, cols = depth.shape 
    gaussian_shifts = np.random.normal(0, 1/2.0, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp
    

if __name__ == "__main__":

    count = 0

    scale_factor = 100 

    while True:

        depth_uint16 = cv2.imread("depth/{}.png".format(count), cv2.IMREAD_UNCHANGED)
        h, w = depth_uint16.shape 

        # Our depth images were scaled by 5000 to store in png format so dividing to get 
        # depth in meters 
        depth = depth_uint16.astype('float') / 5000.0

        depth_interp = add_gaussian_shifts(depth)

        # The depth here needs to converted to cms so scale factor is introduced 
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
        noisy_depth = (35130/np.round(35130/np.round(depth_interp*scale_factor) + np.random.normal(size=(h, w))*(1.0/6.0) + 0.5))/scale_factor 
        noisy_depth = noisy_depth * 5000.0 
        noisy_depth = noisy_depth.astype('uint16')

        # Displaying side by side the orignal depth map and the noisy depth map with barron noise cvpr 2013 model
        cv2.namedWindow('Adding Kinect Noise', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Adding Kinect Noise', np.hstack((depth_uint16, noisy_depth)))
        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        
        count = (count + 1)% 500