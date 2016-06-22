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
__author__ = "Clarence Dillon, Raghav Joshi"
__copyright__ = "Copyright 2016, ICF International inc and U.S. Army Research Laboratories"
__credits__ = ["Clarence Dillon", "Raghav Joshi", "Michael Smith", "Paul Cummings", "Zach Cummings", "Sue Dass",
               "Robert Brusso"]
__license__ = "Mozilla (MPL) Public License"
__version__ = "0.8"
__maintainer__ = "Clarence Dillon"
__email__ = "clarence.dillon@icfi.com"
__status__ = "Beta"

from random import randrange
from copy import copy

import settings
import agents
import data_io


## Initialize the simulation using settings.py parameters
settings.init()

#  Get the relevant GIFT survey to parse
settings.course_name, settings.surveys = data_io.fetchSurvey(settings.DATA_DIR, settings.GIFT_SURVEY)
if settings.verbosity == True:
    print "\nReading in the GIFT survey for " + settings.course_name
    print "\nThere are " + str(len(settings.surveys)) + " surveys available in this course."

## Then parse the survey into answer models. Of course, we have to already know which surveys we want to use. In this
#  case, we want the following:
#  pre-lesson attitude survey at index 0,
#  pre-lesson survey at index 1,
#  post-lesson survey at index 2,
#  (we avoid the post-lesson attitude survey at index 3)
#  reaction survey (after parts 1 and 2) at index 4
#  satisfaction survey (overall) at index 5
settings.survey_models = data_io.parseSurvey(settings.surveys)


## Generate learners (learner profiles) to be saved/retreived from the LRS
settings.learners = [agents.Learner(x) for x in xrange(settings.MIN_STUS)]
data_io.saveJSON(settings.learners, "learners")

## In batches of COURSE_SIZE at a time, learners will start the marksmanship training course in one of several offerings.
#  So, we have to first create course section offerings
num_offers = settings.MIN_STUS / settings.COURSE_SIZE
learner_set = set(copy(settings.learners))

#  Then we assign learners to course sections by creating a new "Course_Offering" object and enrolling them. No need
#  for randomness because the learners are already randomly heterogeneous.
for o in xrange(num_offers):
    offering = settings.course_name + "-" + str(o)
    course = agents.Course_Offering(offering)
    for i in range(0, settings.COURSE_SIZE):
        if len(learner_set) > 0:
            l = learner_set.pop()
            course.enrollment.append(l)
    settings.course_sections.append(course)

# We put the learners through the course, section by section
for co in settings.course_sections:
    for enrolled_learner in co.enrollment:
        enrolled_learner.takeAttitudeSurvey()
        enrolled_learner.takePreLessonSurvey()
        enrolled_learner.takePostLessonSurvey()
        #enrolled_learner.doRecordFireExercise()
        enrolled_learner.takeReactionSurvey()
        enrolled_learner.takeSatisfactionSurvey()





