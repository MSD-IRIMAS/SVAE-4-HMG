import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio as iio
import os

def create_directory(directory_path):
    
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)

def plot_limbs(ax, X_joints, Y_joints, Z_joints):

    # spine base with spine mid
    indexes_of_joints = [0,3]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    # spine mid with spine shoulder
    indexes_of_joints = [3,6]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    indexes_of_joints = [6,9]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    indexes_of_joints = [9,12]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    indexes_of_joints = [12,15]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    # spine base with left hip
    indexes_of_joints = [0,1]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)


    # left hip with left knee
    indexes_of_joints = [1,4]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    # left knee with left anckle
    indexes_of_joints = [4,7]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    # left knee with left anckle
    indexes_of_joints = [7,10]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)
    
    #spine base with right hip
    indexes_of_joints = [0,2]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    # right hip with right knee
    indexes_of_joints = [2,5]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    # right knee with right anckle
    indexes_of_joints = [5,8]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    # right knee with right anckle
    indexes_of_joints = [8,11]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    # spine shoulder with left shoulder
    indexes_of_joints = [9,13]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)


    # left shoulder with left elbow
    indexes_of_joints = [13,16]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    #left elbow with left wrist
    indexes_of_joints = [16,18]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    #left wrist with left hand
    indexes_of_joints = [18,20]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    indexes_of_joints = [20,22]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    # spine shoulder with right shoulder
    indexes_of_joints = [9,14]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)


    # right shoulder with right elbow
    indexes_of_joints = [14,17]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    # right elbow with right wrist
    indexes_of_joints = [17,19]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

    # right wrist with right hand
    indexes_of_joints = [19,21]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)
    
    indexes_of_joints = [21,23]
    ax.plot(X_joints[indexes_of_joints],Y_joints[indexes_of_joints],Z_joints[indexes_of_joints],color='black',lw=2)

def plot_skeleton(x, output_directory='./', title='No Title'):

    create_directory(output_directory + 'plots/')
    create_directory(output_directory + 'gif/')
    
    fig = plt.figure(figsize=(15,20))
    ax = fig.add_subplot(111,projection='3d')
    ax.view_init(elev=10.,azim=-90)

    length_TS = x.shape[0]
    num_joints = x.shape[1]
    dim = x.shape[2]

    images = []

    for t in range(length_TS):

        plt.cla()

        X_joints = x[t].reshape(num_joints, dim)[:,0]
        Y_joints = x[t].reshape(num_joints, dim)[:,2]
        Z_joints = x[t].reshape(num_joints, dim)[:,1]*(-1)

        ax.scatter(X_joints, Y_joints, Z_joints, c='black', depthshade=False, s=100)

        plot_limbs(ax=ax, X_joints=X_joints, Y_joints=Y_joints, Z_joints=Z_joints)

        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(-1,0)

        ax.set_xlabel("x", fontsize=60)
        ax.set_ylabel("y", fontsize=60)
        ax.set_zlabel("z", fontsize=60)

        ax.set_title(title, fontsize=40)

        plt.savefig(output_directory + 'plots/t_'+str(t)+".png")

        images.append(iio.imread(output_directory+'plots/t_'+str(t)+'.png'))

    kwargs = {'duration' : 0.1}

    iio.mimsave(output_directory + 'gif/skeleton.gif', images, 'GIF', **kwargs)