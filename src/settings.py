# This is not a valid entry point.
from loremipsum import Generator
from loremipsum import get_sentences


def init():
    mes = "\nThis Python script is an agent-based simulation to generate learner experiences in U.S. Army marksmanship \n" \
          "training for the express purpose of demonstrating how data collected from user learning/training \n" \
          "experiences can be used to enhance training effectiveness and course evaluation. "
    print(mes)

    # experiment controls
    global debuging
    global verbosity
    global timestamp
    
    # fixed values
    global MIN_STUS
    global COURSE_SIZE
    global DB_CONN
    global DB
    global DATA_DIR
    global GIFT_SURVEY
    
    # shared content
    global course_name
    global surveys
    global learners
    global course_sections
    global survey_models
    global tg
    global sample

    debuging = False
    verbosity = True
    timestamp = None

    # initialize configuration settings and globals
    COURSE_SIZE = 400
    MIN_STUS = 500  # future versions will have this be an input from the webpage
    if MIN_STUS % COURSE_SIZE != 0:
        MIN_STUS = MIN_STUS + (COURSE_SIZE - (MIN_STUS % COURSE_SIZE))

    # Data Configuration for data_io file operations
    DATA_DIR = "../newData/"
    DB = ""
    DB_CONN = ""

    GIFT_SURVEY = "Domain/Marksmanship Course/ModifiedMarksmanshipCourse.course.surveys.export"
    # IPSUM = DATA_DIR + "reaction/filleratti-ipsum.txt"
    # getIpsum(IPSUM)

    course_name = ""
    surveys = []
    learners = []
    course_sections = []
    survey_models = []


def getIpsum(the_text):
    global sample
    with open(the_text, 'r') as sample_txt:
        sample = sample_txt.read()
        return sample
    
       

