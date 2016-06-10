#!/usr/bin/env python
'''
This Python script is an agent-based simulation to generate learner experiences in U.S. Army marksmanship training for
the express purpose of demonstrating how data collected from user learning/training experiences can be used to enhance
training effectiveness and course evaluation.

The intent of this script it to read a GIFT (see gifttutoring.org) survey and generate agent-specific responses, then
simulate gateway exercise/assessments as a practicum. The included case simulates basic rifle marksmanship (BRM)
training and follows the data collection pattern that Problem Solutions uses in their REAPER experiments.

Other scripts associated with this simulation will read and analyze data generated by this simulation. In real life,
that data will be stored to a learning record store (LRS). For the sake of simplicity, this script stores data in JSON
documents.
'''
__author__ = "Clrence Dillon, Raghav Joshi"
__copyright__ = "Copyright 2016, ICF International inc and U.S. Army Research Laboratories"
__credits__ = ["Clarence Dillon", "Raghav Joshi", "Michael Smith", "Paul Cummings", "Zach Cummings", "Sue Dass",
               "Robert Brusso"]
__license__ = "Mozilla (MPL) Public License"
__version__ = "2.0"
__maintainer__ = "Clarence Dillon"
__email__ = "clarence.dillon@icfi.com"
__status__ = "Beta"

import settings
import future_builtins




## Get the minimum number of learners to generate in this simulation. If a course offering is too small, the simulation
#  will generate more to round out the cadre.
MIN_NUM_STUS = 1000 #this should become a user input value through the GUI. The

## Generate learners (learner profiles) to be saved/retreived from the LRS


## In batches of 20 - 40 at a time, learners will start the marksmanship training course in one of several offerings.

## They start with an attitude survey


## Then take a pre-lesson knowledge assessment


## Learners do computer-based lesson, then take a post-lesson knowledge assessment


##
