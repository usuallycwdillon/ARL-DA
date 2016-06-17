## This is not a valid entry point. See learner_xSim.py

import settings

survey = settings.GIFT_SURVEY


settings.course_name = survey['Survey_Context_Name']

settings.surveys = survey['Survey_Context_Surveys'][0]['Survey_Context_Survey_Survey']['Survey_Pages']