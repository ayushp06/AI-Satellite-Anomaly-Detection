import numpy as np

#This function performs quaternion multiplication, written as q * p
def qmul(q, p):
    w, x, y, z = q
    W, X, Y, Z = p

    #Below is the Hamilton Product Formula that combines two rotations together
    return np.array([
        w*W - x*X - y*Y - z*Z,
        w*X + x*W + y*Z - z*Y,
        w*Y - x*Z + y*W + z*X,
        w*Z + x*Y - y*X + z*W
    ], dtype = float)
    
#This function normalizes the quaternion so the rotation is properly represented
def qnorm(q):
    return q/np.linalg.norm(q)

#This function computes the time derivative of the quaternion, given angular velocity
def dqdt(q, w_b):
    return 0.5 * qmul(q, np.hstack([0.0, w_b]))

#Implementing core physics with torque and angular velocity so the body can accelerate or tumble
def attitudeStep(q, w_b, t_b, I, dt):
    Iw = I @ w_b
    wDot = np.linalg.solve(I, (t_b - np.cross(w_b, Iw)))
    
    #Semi implicit Euler: update w first, then q using new w
    wNew = w_b + dt * wDot
    qNew = qnorm(q + dt * dqdt(q, wNew))
    return qNew, wNew
