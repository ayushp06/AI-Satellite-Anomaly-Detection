#Initializes time, quaternion, and angular velocity to be logged
def teleBuild(t, q, w, faultFlag = False):
    return {
        "t" : float(t),
        "q" : [float(x) for x in q],
        "w" : [float(val) for val in w],
        "fault" : int(bool(faultFlag))
    }