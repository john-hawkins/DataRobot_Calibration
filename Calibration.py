# #######################################################################################
# Methods for re-calibrating the probability outputs of a DataRobot model
# #######################################################################################

import itertools 
import datarobot as dr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn.metrics import (brier_score_loss, precision_score, recall_score)
from sklearn.linear_model import LogisticRegression

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
#isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')
#sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

# #######################################################################################
#  GENERATE A CALIBRATION MODEL
#  NOTES: The content of actual_col must be numeric
#         I am wrapping it up in a dictionary with the maximum observed because for some
#         reason the IR will return NaN for probabilities outside the bounds of the 
#         training data
# #######################################################################################
def generate_calibration_model( df, pred_col, actual_col ):
    ir = IsotonicRegression()
    y_ = ir.fit_transform( df[pred_col], df[actual_col] )
    calib = {}
    calib['method'] = 'ir'
    calib['mod'] = ir
    calib['max_obs'] = max( df[pred_col] ) 
    calib['max_cal'] = max( y_ ) 
    return calib

def generate_calibration_model_platt( df, pred_col, actual_col ):
    lr = LogisticRegression()                                                   
    temp = np.reshape(df[pred_col], (-1, 1)) # LR needs X to be 2-dimensional
    lr.fit(  temp, df[actual_col] )     
    y_ = lr.predict_proba( temp )[:,1]
    calib = {}
    calib['method'] = 'platt'
    calib['mod'] = lr
    calib['max_obs'] = max( df[pred_col] )
    calib['max_cal'] = max( y_ )
    return calib


#df_calibrate['calibrated'] =  ir.transform( df_calibrate[pred_col] )
#max_scored = max(df_calibrate['pred_col'])
#max_calib = max(df_calibrate['calibrated'])

# #######################################################################################
#  APPLY THE CALIBRATION 
# #######################################################################################
def apply_calibration( data, calib ) :
    temp = data.copy()
    temp[ temp > calib['max_obs'] ] = calib['max_obs']
    if calib['method'] == 'platt':
        temp = np.reshape( data, (-1, 1)) # LR needs X to be 2-dimensional
        calibrated = calib['mod'].predict_proba( temp )[:,1]
    else:
        calibrated =  calib['mod'].transform( temp )
    return calibrated

# #######################################################################################
#  GENERATE A RELIABILITY PLOT TO COMPARE THE CALIBRATION OF SCORES
# #######################################################################################
def generate_reliability_plot( df, pred_cols, actual_col ):
    fig = plt.figure()
    cols = ['r.', 'g.', 'b.', 'c.', 'm.', 'k.', 'y.']
    index = 0
    for pred_col in pred_cols:
        #print(pred_col)
        temp = df.sort_values(by=[pred_col]).copy()
        temp['group'] = round( temp[pred_col]*50 ) / 50
        plotdf = temp.groupby('group')[actual_col].mean()
        plt.plot(plotdf, cols[index], markersize=12)
        index = index + 1 
    plt.legend(pred_cols, loc='upper left')
    plt.title('Reliability Plot')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Empirical Probability')
    return plt


# #######################################################################################
#  GENERATE A RELIABILITY PLOT WITH DENSITY
# #######################################################################################
def plot_calibration_curves( df, pred_cols, actual_col, pred_names, fig_index ):
    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for (nam, col) in zip(pred_names, pred_cols):
        frac_pos, mean_pred_value = calibration_curve(df[actual_col], df[col], n_bins=10)
        clf_score = brier_score_loss(df[actual_col], df[col], pos_label=1)
        ax1.plot(mean_pred_value, frac_pos, "s-", label="%s (%1.3f)" % (nam, clf_score))
        ax2.hist(df[col], range=(0, 1), bins=10, label=nam, histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots (reliability curve)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
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
    
 
