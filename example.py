# Import the file: Calibration.py
import Calibration as clb
import datarobot as dr
import pandas as pd

# LOAD THE EXAMPLE DATA THAT WILL BE USED
df_calibrate = pd.read_csv('data/validation_data_with_predictions.csv')
df_forecast = pd.read_csv('data/holdout_data_with_predictions.csv')
df_final_forecast = pd.read_csv('data/final_holdout_with_preds.csv')

# DEFINE THE TWO COLUMNS WE WILL USE
pred_col = "prediction"
actual_col = 'target'

# CREATE THE CALIBRATION MODEL USING THE OUT-OF-FOLD PREDICTIONS
# ON THE INITIAL FIVE FOLD CROSS VALIDATION USING PLATT SCALING
calib = clb.generate_calibration_model_platt(df_calibrate, pred_col, actual_col)

# APPLY THE CALIBRATION TO THE HOLDOUT AND PLOT IT
# COMPARE WITH ORGINAL MODEL AND A MODEL RETRAINED TO USE THE VALIDATION DATA AS WELL
df_forecast['calibrated'] = clb.apply_calibration( df_forecast[pred_col], calib ) 
df_forecast['retrained'] = df_final_forecast[pred_col]
plt = clb.plot_calibration_curves(df_forecast, ['prediction', 'calibrated', 'retrained'], actual_col, ['Original Model', 'Calibrated Model', 'Retrained Model'], 1)
plt.savefig('results/calibration.png')

 
