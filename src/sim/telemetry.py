#Initializes time, quaternion, and angular velocity to be logged
def teleBuild(t, q, w):
    return {
        "t" : round(float(t), 3),
        "q" : [float(x) for x in q],
        "w" : [float(val) for val in w]
    }