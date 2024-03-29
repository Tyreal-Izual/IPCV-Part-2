'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse


'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True


def detect_circles(image, dp=1.2, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=50):
    '''
    Detect circles in an image using HoughCircles method

    Parameters:
    image: Grayscale image where circles are detected.
    dp: Inverse ratio of the accumulator resolution to the image resolution.
    minDist: Minimum distance between the centers of the detected circles.
    param1: Higher threshold for the Canny edge detector.
    param2: Accumulator threshold for the circle centers at the detection stage.
    minRadius: Minimum circle radius.
    maxRadius: Maximum circle radius.

    Returns:
    circles: Detected circles, Nx3 numpy array with x, y, and radius for each circle.
    '''
    # Convert to grayscale if the image is colored
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    image = cv2.GaussianBlur(image, (9, 9), 2)

    # Detect circles
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    return circles



# print("here", flush=True)
if __name__ == '__main__': 

    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', dest='num', type=int, default=6, 
                        help='number of spheres')    
    parser.add_argument('--sph_rad_min', dest='sph_rad_min', type=int, default=10, 
                        help='min sphere  radius x10')
    parser.add_argument('--sph_rad_max', dest='sph_rad_max', type=int, default=16, 
                        help='max sphere  radius x10')
    parser.add_argument('--sph_sep_min', dest='sph_sep_min', type=int, default=4, 
                       help='min sphere  separation')
    parser.add_argument('--sph_sep_max', dest='sph_sep_max', type=int, default=8, 
                       help='max sphere  separation')
    parser.add_argument('--display_centre', dest='bCentre', action='store_true',
                        help='open up another visualiser to visualise centres')
    parser.add_argument('--coords', dest='bCoords', action='store_true')

    parser.add_argument('--display_spheres', dest='bSpheres', action='store_true',
                        help='Display epipolar lines in the 3D visualizer')

    parser.add_argument('--display_centers', dest='bDisplayCenters', action='store_true',
                        help='Display estimated and ground truth centers in the 3D visualizer')

    # Add arguments for noise levels
    parser.add_argument('--pos_noise', dest='position_noise_level', type=float, default=0,
                        help='Noise level for position')
    parser.add_argument('--orient_noise', dest='orientation_noise_level', type=float, default=0,
                        help='Noise level for orientation')

    args = parser.parse_args()

    if args.num<=0:
        print('invalidnumber of spheres')
        exit()

    if args.sph_rad_min>=args.sph_rad_max or args.sph_rad_min<=0:
        print('invalid max and min sphere radii')
        exit()
    	
    if args.sph_sep_min>=args.sph_sep_max or args.sph_sep_min<=0:
        print('invalid max and min sphere separation')
        exit()
	
    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    prev_loc = []
    GT_cents, GT_rads = [], []
    for i in range(args.num):
        # add sphere name
        name_list.append(f'sphere_{i}')

        # create sphere with random radius
        size = random.randrange(args.sph_rad_min, args.sph_rad_max, 2)/10
        sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0., 0.5, 0.5])

        # create random sphere location
        step = random.randrange(args.sph_sep_min,args.sph_sep_max,1)
        x = random.randrange(-h/2+2, h/2-2, step)
        z = random.randrange(-w/2+2, w/2-2, step)
        while check_dup_locations(x, z, prev_loc):
            x = random.randrange(-h/2+2, h/2-2, step)
            z = random.randrange(-w/2+2, w/2-2, step)
        prev_loc.append((x, z))

        GT_cents.append(np.array([x, size, z, 1.]))
        GT_rads.append(size)
        sph_H = np.array(
                    [[1, 0, 0, x],
                     [0, 1, 0, size],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]
                )
        H_list.append(sph_H)

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if args.bCoords:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * (80+random.uniform(-10, 10))/180.
    H1_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H1_0)
    render_list = [(H0_wc, 'view0.png', 'depth0.png'), 
                   (H1_wc, 'view1.png', 'depth1.png')]

#####################################################
    # NOTE: This section relates to rendering scenes in Open3D, details are not
    # critical to understanding the lab, but feel free to read Open3D docs
    # to understand how it works.

    # set up camera intrinsic matrix needed for rendering in Open3D
    img_width=640
    img_height=480
    f=415 # focal length
    # image centre in pixel coordinates
    ox=img_width/2-0.5
    oy=img_height/2-0.5
    K = o3d.camera.PinholeCameraIntrinsic(img_width,img_height,f,f,ox,oy)

    # Here I moved the following original Rendering Code for 3-D  window, to the bottom of the main function. To makesure every rendering that I wrote below in the tasks is visiable in the 3D screen.

    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for (H_wc, name, dname) in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    vis.run()
    vis.destroy_window()
##################################################

    # load in the images for post processings
    img0 = cv2.imread('view0.png', -1)
    dep0 = cv2.imread('depth0.png', -1)
    img1 = cv2.imread('view1.png', -1)
    dep1 = cv2.imread('depth1.png', -1)

    # visualise sphere centres
    pcd_GTcents = o3d.geometry.PointCloud()
    pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
    pcd_GTcents.paint_uniform_color([1., 0., 0.])
    if args.bCentre:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()


    ###################################
    '''
    Task 3: Circle detection
    Hint: use cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Write your code here
    '''
    # Detect circles in both images
    circles0 = detect_circles(img0)
    circles1 = detect_circles(img1)
    detected_centers0 = []
    detected_centers1 = []
    # Process and display the results
    if circles0 is not None:
        circles0 = np.uint16(np.around(circles0))
        for i in circles0[0, :]:
            # Draw the outer circle
            cv2.circle(img0, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(img0, (i[0], i[1]), 2, (0, 0, 255), 3)
            detected_centers0.append(np.array([i[0], i[1], 1]))

    if circles1 is not None:
        circles1 = np.uint16(np.around(circles1))
        for i in circles1[0, :]:
            # Draw the outer circle
            cv2.circle(img1, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(img1, (i[0], i[1]), 2, (0, 0, 255), 3)
            detected_centers1.append(np.array([i[0], i[1], 1]))

    # Save the images with the drawn circles
    cv2.imwrite('hough_circles_img0.png', img0)
    cv2.imwrite('hough_circles_img1.png', img1)

    ###################################
    ###################################
    '''
    Task 10: Investigate impact of noise added to relative pose

    Write your code here:
    '''

    # Introduce noise to the camera poses
    # Adding position noise
    position_noise_H0 = np.random.normal(0, args.position_noise_level, 3)
    position_noise_H1 = np.random.normal(0, args.position_noise_level, 3)
    H0_wc[:3, 3] += position_noise_H0
    H1_wc[:3, 3] += position_noise_H1

    # Adding orientation noise
    orientation_noise_H0 = np.random.normal(0, args.orientation_noise_level, (3, 3))
    orientation_noise_H1 = np.random.normal(0, args.orientation_noise_level, (3, 3))
    H0_wc[:3, :3] += orientation_noise_H0
    H1_wc[:3, :3] += orientation_noise_H1

    # Normalize to maintain orthogonality of rotation matrices
    H0_wc[:3, :3] /= np.linalg.norm(H0_wc[:3, :3], axis=0)
    H1_wc[:3, :3] /= np.linalg.norm(H1_wc[:3, :3], axis=0)
    ###################################

    ###################################
    '''
    Task 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
          Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    
    Write your code here
    '''
    Mpi=np.linalg.inv(K.intrinsic_matrix)
    # now compute matrix which converts iamge coordinates to pixel coordinates

    H_10 = np.matmul(H0_wc, np.linalg.inv(H1_wc))

    # this gives P0=R10*P1+T10
    # hence P1=R10^T(P0-T10) which is in the same the form
    # hence
    R = H_10[:3, :3].T # R10^T
    T = H_10[:3, 3] # T10

    # now form the cross pdt matrix S from T
    S = np.array([
        [0, -T[2], T[1]],
        [T[2], 0, -T[0]],
        [-T[1], T[0], 0]
    ])

    # and hence the essential matrix
    E = np.matmul(R, S)
    print('Essential Matrix')
    print(E)

    # now use the pixel to image coordinate matrix to compute the fundamental matrix
    F = np.matmul(np.matmul(Mpi.T, E), Mpi)
    print('Fundamental Matrix:')
    print(F)
    # detected_centers0 = [circle[:2] for circle in circles0[0, :]]
    # detected_centers1 = [circle[:2] for circle in circles1[0, :]]

    epipolar_lines = []
    img = cv2.imread('hough_circles_img1.png')
    for pt1_img0 in detected_centers0:

        # Calculate corresponding epipolar line in image1
        # print(pt1_img0)
        u = np.matmul(F, pt1_img0)
        a, b, c = u
        m = -a/b
        c = -c/b

        # Compute intersections of the epipolar line with the image borders
        p0 = np.array([0, m*0+c]).astype(int)
        p1 = np.array([img_width, m*img_width+c]).astype(int)
        # print(p0, p1)
        # Store the epipolar line points
        epipolar_lines.append((p0, p1))

        # Draw the line in image1
        img = cv2.line(img, p0, p1, (255, 0, 0), 1)

    cv2.imwrite('view1_eline_fmat.png', img)
    ###################################


    ###################################
    '''
    Task 5: Find correspondences

    Write your code here
    '''
    correspondences = []

    for center0, (p0, p1) in zip(detected_centers0, epipolar_lines):
        # Create an epipolar line equation: ax + by + c = 0
        # p0 and p1 are two points on the line
        line = np.cross(np.append(p0, 1), np.append(p1, 1))

        min_distance = float('inf')
        corresponding_center = None

        for center1 in detected_centers1:
            # Convert center to homogeneous coordinates
            point = np.append(center1[:2], 1)

            # Check the epipolar constraint: P_R^T F P_L = 0
            # In this case, line is the equivalent of P_R^T F
            distance = abs(np.dot(line, point)) / np.linalg.norm(line[:2])

            # Find the point with minimum distance to the epipolar line
            if distance < min_distance:
                min_distance = distance
                corresponding_center = center1

        # If a corresponding center is found, store the correspondence
        if corresponding_center is not None:
            correspondences.append((center0, corresponding_center))
            print(f"Correspondence found: Image 0 Center: {center0}, Image 1 Center: {corresponding_center}")
    print("\nAll Correspondences:")
    for corr in correspondences:
        print(f"Image 0 Center: {corr[0]}, Image 1 Center: {corr[1]}")
        ###################################


    ###################################
    '''
    Task 6: 3-D locations of sphere centres

    Write your code here
    '''
    # K is the Open3D PinholeCameraIntrinsic object

    # Extract the 3x3 intrinsic matrix from the Open3D PinholeCameraIntrinsic object
    K_matrix = np.array(K.intrinsic_matrix).reshape((3, 3))

    R = H_10[:3, :3].T # R10^T
    T = H_10[:3, 3] # T10
    T = -T #view ->reference to reference->view
    reconstructed_centers = []
   
    for (pL, pR) in correspondences:  # Loop through the good pairs obtained from Task 5
        # print("pl",pL)
        # print("pr",pR)
        # Convert image points to normalized device coordinates
        pL_n = np.linalg.inv(K_matrix) @ np.array([pL[0], pL[1], 1])
        pR_n = np.linalg.inv(K_matrix) @ np.array([pR[0], pR[1], 1])
        # print(pL_n)
        # print("pR_n", pR_n)
        # print("R",R)
        # print("R.T",R.T)

    # Compute cross product
        pL_cross_pR = np.cross(pL_n, np.matmul(R.T, pR_n))

        # Form the linear system Ax = 0
        A = np.column_stack([pL_n, -np.matmul(R.T, pR_n), -pL_cross_pR])
        # print("T",T)

        b = -T  # Make sure T is a column vector

        # Solve the linear system using least squares
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # a, b, and c are the components of the solution x
        a, b, c = x.flatten()  # Flatten in case x is a 2D array
        # print("a,b,c",a,b,c)
        # Reconstruct the 3D point
        X = a * pL_n - c * pL_cross_pR
        # print("X: ", X)
        X = np.append(X, 1)
        X = np.dot(np.linalg.inv(H0_wc), X)
       # X = (a*pL_n + b*np.matmul(R.T, pR_n) + T)/2
       # Append the reconstructed 3D center to the list
        reconstructed_centers.append(X[:3])

    # Print or process the reconstructed centers
    for center in reconstructed_centers:
        print("Reconstructed 3D center:", center)

    ###################################


    ###################################
    '''
    Task 7: Evaluate and Display the centres

    Write your code here
    '''
    # Create a point cloud for estimated sphere centres
    pcd_estimated_cents = o3d.geometry.PointCloud()
    pcd_estimated_cents.points = o3d.utility.Vector3dVector(np.array(reconstructed_centers))
    pcd_estimated_cents.paint_uniform_color([0., 0., 1.])  # Blue color for estimated centres

    # Add both point clouds (GT and estimated) to the visualizer
    if args.bDisplayCenters:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents, pcd_estimated_cents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()
    # print("GT_cents",GT_cents)
    # print("reconstructed_centers",reconstructed_centers)
    
    # Match the estimated centers with their corresponding ground truth centers
    # Use the correspondences from Task 5
    matched_estimated_centers = [None] * len(GT_cents)
    for i, (center0, _) in enumerate(correspondences):
        matched_estimated_centers[i] = reconstructed_centers[i]
    # print("matched_estimated_centers", matched_estimated_centers)

    # Compute errors in sphere centre estimates by matching each estimated center
    # with the closest ground truth center
    errors = []
    for est_center in reconstructed_centers:
        closest_gt_center = min(GT_cents, key=lambda gt_center: np.linalg.norm(np.array(gt_center[:3]) - np.array(est_center)))
        error = np.linalg.norm(np.array(closest_gt_center[:3]) - np.array(est_center))
        errors.append(error)


    # Print or process the errors
    for error in errors:
        print("Error in estimated center:", error)

    ###################################


    ###################################
    '''
    Task 8: 3-D radius of spheres

    Write your code here
    '''
    estimated_radii = []

    for i in range(len(reconstructed_centers)):
        # Calculate the distance from the camera to the sphere center
        d0 = np.linalg.norm(reconstructed_centers[i] - H0_wc[:3, 3])

        # Apparent radii from images
        R_2D = circles0[0][i][2]  # radius from image 1

        # Calculate real radius using similar triangles
        R_3D = R_2D * d0 / f

        estimated_radii.append(R_3D)

        print(f"Estimated 3D radius for sphere {i}: {R_3D}")
    ###################################


    ###################################
    '''
    Task 9: Display the spheres

    Write your code here:
    '''
    # Compute radius errors
    radius_errors = []
    for gt_rad, est_rad in zip(GT_rads, estimated_radii):
        error = abs(gt_rad - est_rad)
        radius_errors.append(error)
        print(f"Radius Error: {error}")

    # Visualization
    if args.bSpheres:  # Check if the argument to display spheres is set
        gt_spheres = []
        est_spheres = []

        for gt_center, gt_rad, est_center, est_rad in zip(GT_cents, GT_rads, reconstructed_centers, estimated_radii):
            # Ground Truth Sphere
            gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=gt_rad)
            gt_sphere.translate(gt_center[:3])
            gt_sphere.paint_uniform_color([1, 0, 0])  # Red color for ground truth
            gt_spheres.append(gt_sphere)

            # Estimated Sphere
            est_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=est_rad)
            est_sphere.translate(est_center[:3])
            est_sphere.paint_uniform_color([0, 0, 1])  # Blue color for estimated
            est_sphere.compute_vertex_normals()
            est_spheres.append(est_sphere)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for sphere in  gt_spheres + est_spheres:
            vis.add_geometry(sphere)
        vis.run()
        vis.destroy_window()
    ###################################

    ###################################
    '''
    Task 10: Investigate impact of noise added to relative pose

    Write your code here:
    '''
    ##################################
