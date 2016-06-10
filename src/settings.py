def init():
    mes = "This Python script is an agent-based simulation to generate learner experiences in U.S. Army marksmanship \n" \
          " training for the express purpose of demonstrating how data collected from user learning/training \n" \
          "experiences can be used to enhance training effectiveness and course evaluation. "
    print(mes)

    # experiment controls
    global debuging
    global verbosity
    global timestamp

    global MIN_STUS

    debuging = False
    verbosity = True
    timestamp = None

    # initialize configuration settings and globals
    MIN_STUS = 1000  # future versions will have this be an input from the webpage


def getTimeStamp(detail = 'short'):
    global timestamp
    timestamp = dtg.datetime.now()
    dt = str("%02d"%timestamp.hour) + str("%02d"%timestamp.minute) + "." + str("%02d"%timestamp.second) + "EDT"
    if detail != 'short':
        dt += "_" + str(timestamp.year) + str("%02d"%timestamp.month) + str("%02d"%timestamp.day)
    return dt
