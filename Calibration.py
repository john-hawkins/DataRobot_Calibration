# #######################################################################################
# Methods for re-calibrating the probability outputs of a DataRobot model
# #######################################################################################

import datarobot as dr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state



# #######################################################################################
#  GENERATE A CALIBRATION MODEL
#  NOTE: The content of actual_col must be numeric
# #######################################################################################
def generate_calibration_model(df, pred_col, actual_col):
    ir = IsotonicRegression()
    y_ = ir.fit_transform(df[pred_col], df[actual_col])
    return ir


# #######################################################################################
#  GENERATE A RELIABILITY PLOT TO COMPARE THE CALIBRATION OF SCORES
# #######################################################################################
def generate_reliability_plot(df, pred_cols, actual_col):
    fig = plt.figure()
    cols = ['r.', 'g.-', 'b-', 'c.', 'm.', 'k.', 'y.']
    index = 0
    for pred_col in pred_cols:
        #print(pred_col)
        temp = df.sort_values(by=[pred_col]).copy()
        temp['group'] = (round(temp[pred_col]*50)*2)/100
        plotdf = temp.groupby('group')[actual_col].mean()
        plt.plot(plotdf, cols[index], markersize=12)
        index = index + 1
    plt.legend(pred_cols, loc='upper left')
    plt.title('Reliability Plot')
    return plt



# #######################################################################################
#  GENERATE A CALIBRATION MODEL FROM DATAROBOT PROJECT
#  -- THIS IS INCOMPLETE BECAUSE THE DATAROBOT API DOES NOT ALLOW ME TO EXPORT THE TARGET
#     VALUES WITH THE SCORES
#
# #######################################################################################
def generate_calibrator( project_id, model_id ):
    # Extract the scores for the validation/backtest and holdout data. 
    project = dr.Project.get(project_id=project_id)
    model = dr.Model.get(project_id, model_id)

    # ONLY BINARY CLASSIFICATION IS SUPPORTED
    if (project.target_type == 'binary'):
        print('Binary Classification Only')
        exit(0)

    # UNLOCK THE HOLDOUT IF NECESSARY
    if project.holdout_unlocked == False :
        project.unlock_holdout()

    if (project.partition['cv_method'] == 'datetime'):
        pred_job = model.request_training_predictions( dr.enums.DATA_SUBSET.ALL_BACKTESTS )
        preds = pred_job.get_result_when_complete()
        df_calibrate = preds.get_all_as_dataframe()
        pred_job = model.request_training_predictions( dr.enums.DATA_SUBSET.HOLDOUT )
        preds = pred_job.get_result_when_complete()
        df_forecast = preds.get_all_as_dataframe()
    else:
        #pred_job = model.request_training_predictions( dr.enums.DATA_SUBSET.HOLDOUT )
        #preds = pred_job.get_result_when_complete()
        #df_forecast = preds.get_all_as_dataframe()
        pred_job = model.request_training_predictions( dr.enums.DATA_SUBSET.VALIDATION_AND_HOLDOUT )
        preds = pred_job.get_result_when_complete()
        dataframe = preds.get_all_as_dataframe()
        df_calibrate = dataframe[ dataframe['partition_id'] == '0.0']
        df_forecast = dataframe[ dataframe['partition_id'] == 'Holdout']
    
 
