## This is not a valid entry point. See learner_xSim.py

import settings
import json
import jsonpickle
from random import choice
import datetime as dtg
from loremipsum import get_sentences
from loremipsum import Generator
from time import gmtime, strftime


def saveJSON(content, location, name=None):
    data = content
    dir = location + "/"
    for d in data:
        if location == "learners":
            name = d.learner_id
            fn = settings.DATA_DIR + dir + name
            frozen = jsonpickle.encode(d)
            with open(fn, 'w') as outfile:
                outfile.write(unicode(frozen))
                fn.close()
        if location == "results":
            print "Saving results..."
            name = d['explanandum'] + ".json"
            fn = settings.DATA_DIR + dir + name
            with open(fn, 'w') as f:
                json.dump(d, f)


def fetchSurvey(dir, file):
    survey_file = dir + file
    survey = open(survey_file).read()
    survey_json = json.loads(survey)
    course_name = survey_json['Survey_Context_Name']
    surveys = survey_json['Survey_Context_Surveys'][0]['Survey_Context_Survey_Survey']['Survey_Pages']
    return course_name, surveys


def parseSurvey(survey):
    '''
    Method to parse the survey into dicts that represent model answers for each learner to fill-in and store.
    :param survey:
    :return: Nothing. Sets global survey_models
    '''
    survey_models = [getSurveyData(s) for s in survey]
    return survey_models


def getSurveyData(pages):
    page_elements = pages['Elements']
    data_objects = []
    for each_element in page_elements:
        if each_element['Question']['Type'] == 'MultipleChoice':
            data_objects.append(getElementData(each_element))
        elif each_element['Question']['Type'] == 'MatrixOfChoices':
            data_objects.append(getMatrixData(each_element))
        elif each_element['Question']['Type'] == 'FillInTheBlank':
            data_objects.append(getBlankFiller(each_element))

    survey_response = {"Survey_Name": pages['Survey_Page_Name'],
                       "learner_id": "",
                       "date_time": "",
                       "Survey_Objects": data_objects
                       }
    return survey_response


def getElementData(page_elements):
    each_element = page_elements
    scoring = each_element['Properties']['Answer_Weights']['Property_Value']['StringPayload']
    scoring = scoring.split(',')
    scoring = [int(s) for s in scoring]
    survey_object = {"Question": each_element['Question']['Question_Text'],
                     "Type": each_element['Question']['Type'],
                     "Answer": [0, 0, 0, 0],
                     "Scoring": scoring,
                     "Points": 0
                     }
    return survey_object


def getMatrixData(page_elements):
    each_element = page_elements
    question_cols = []
    question_rows = []
    col_labels = each_element['Question']['Properties']['Column_Options']['Property_Value']['List_Options']
    row_labels = each_element['Question']['Properties']['Row_Options']['Property_Value']['List_Options']
    answer_wts = each_element['Properties']['Answer_Weights']['Property_Value']['REPLY_WEIGHTS']

    if len(answer_wts) != len(row_labels):
        print "Wet cleanup on isle 3. data_io.getMatrixData is getting garbage from the survey."

    if len(each_element['Question']['Categories']) > 0:
        category = each_element['Question']['Categories'][0]
    else: category = "Not Categorized"

    for each_option in col_labels:
        question_cols.append(each_option['Text'])

    for each_option in row_labels:
        question_rows.append(each_option['Text'])

    answer_index = [0] * len(question_rows)

    survey_object = {"Cols": question_cols,
                     "Rows": question_rows,
                     "Weights": answer_wts[0],
                     "Answers": answer_index,
                     "Category": category,
                     "Type": each_element['Question']['Type']}
    return survey_object


def getBlankFiller(page_elements):
    each_element = page_elements
    # This text is completely random, so we can just fill it in right away
    ans_len = choice([2, 3, 4, 5, 6])
    # random_text = settings.tg.generate_sentences(ans_len)
    random_text = ' '.join(get_sentences(ans_len))
    # print random_text
    survey_object = {"Question": each_element['Question']['Question_Text'],
                     "Type": each_element['Question']['Type'],
                     "Answer": random_text
                     }
    return survey_object


def getTimeStamp(detail = 'short'):
    global timestamp
    timestamp = dtg.datetime.now()
    dt = str("%02d"%timestamp.hour) + str("%02d"%timestamp.minute) + "." + str("%02d"%timestamp.second) + "EDT"
    if detail != 'short':
        dt += "_" + str(timestamp.year) + str("%02d"%timestamp.month) + str("%02d"%timestamp.day)
    return dt
