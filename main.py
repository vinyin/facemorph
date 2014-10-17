from scipy.misc import imread, imsave
from scipy.interpolate import interp2d
import numpy as np
from scipy.spatial import Delaunay
import os

def read_face(filename):
    face = imread(filename+".jpg")
    facepts = []
    facepts.append([0,0])
    facepts.append([face.shape[0],0])
    facepts.append([0,face.shape[1]])
    facepts.append([face.shape[0],face.shape[1]])
    with open(filename+".asf") as pointfile:
        for line in pointfile:
            l = line.split()
            if "#" in l:
                continue
            if len(l) > 4:
                facepts.append([float(l[3])*face.shape[0],float(l[2])*face.shape[1]])

    return face, np.array(facepts)
    
def pt_in_tri(pt, tri):
    #computed using barycentric coordinate formula from http://en.wikipedia.org/wiki/Barycentric_coordinate_system
    d = (tri[1][1] - tri[2][1])*(tri[0][0] - tri[2][0]) + (tri[2][0] - tri[1][0])*(tri[0][1] - tri[2][1])
    s = ((tri[1][1] - tri[2][1])*(pt[0] - tri[2][0]) + (tri[2][0] - tri[1][0])*(pt[1] - tri[2][1]))/d
    t = ((tri[2][1] - tri[0][1])*(pt[0] - tri[2][0]) + (tri[0][0] - tri[2][0])*(pt[1] - tri[2][1]))/d
    return (s >= 0) and (t >= 0) and (s + t <= 1)
    
def tsearch(pt, tri):
    for i in range(len(tri)):
        if pt_in_tri(pt, tri[i]):
            return i
    return None
    
def compute_transform(t1, t2):
    A = np.linalg.inv(np.array([[t1[0][1],t1[0][0],1],
                                [t1[1][1],t1[1][0],1],
                                [t1[2][1],t1[2][0],1]]))
    x1 = np.array([t2[0][0],t2[1][0],t2[2][0]])
    x2 = np.array([t2[0][1],t2[1][1],t2[2][1]])
    b1 = np.dot(A, x2)
    b2 = np.dot(A, x1)
    return np.array([b1,b2,[0, 0, 1]])

def morph(face1, face1pts, face2, face2pts, warp_frac=0.5, dissolve_frac=0.5):
    #we assume face1pts and face2pts contains the corners of the image
    face1 = np.copy(face1)
    face2 = np.copy(face2)
    avgpts = (1-warp_frac)*face1pts + warp_frac*face2pts
    avgpts[:4] = np.trunc(avgpts[:4])
    face = np.zeros((avgpts[3,0], avgpts[3,1], 3))
    delaunay_triangulation = Delaunay(avgpts)
    simplices = delaunay_triangulation.simplices
    triang1 = [[face1pts[s[0]], face1pts[s[1]], face1pts[s[2]]] for s in simplices]
    triang2 = [[face2pts[s[0]], face2pts[s[1]], face2pts[s[2]]] for s in simplices]
    triang = [[avgpts[s[0]], avgpts[s[1]], avgpts[s[2]]] for s in simplices]
    affine_t1 = [compute_transform(triang[i], triang1[i]) for i in range(len(triang1))]
    affine_t2 = [compute_transform(triang[i], triang2[i]) for i in range(len(triang2))]
    for y in range(face.shape[0]):
        for x in range(face.shape[1]):
            trinum1 = tsearch((y, x), triang1)
            vec1 = np.dot(affine_t1[trinum1], np.array([x, y, 1]))
            vec1 = np.trunc(vec1)
            trinum2 = tsearch((y, x), triang2)
            vec2 = np.dot(affine_t2[trinum2], np.array([x, y, 1]))
            vec2 = np.trunc(vec2)
            try:
                face[y,x,:] = (1-dissolve_frac)*face1[vec1[1]-1,vec1[0]-1] + dissolve_frac*face2[vec2[1]-1,vec2[0]-1]
            except:
                print (vec1[1]-1,vec1[0]-1), (vec2[1]-1,vec2[0]-1)
    return face

facedb = ["38-1m", "34-1m"]
f1, f1p = read_face(facedb[0])
f2, f2p = read_face(facedb[1])
mf = morph(f1, f1p, f2, f2p)
imsave("morphedface.jpg", mf)