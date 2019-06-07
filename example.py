# Import the file: Calibration.py
import Calibration as clb
import datarobot as dr
import pandas as pd

# LOAD THE SAMPLE DATA THAT WILL BE USED
df_calibrate = pd.read_csv('data/validation_data_with_predictions.csv')
df_forecast = pd.read_csv('data/holdout_data_with_predictions.csv')
df_final_forecast = pd.read_csv('data/final_holdout_with_preds.csv')

# DEFINE THE TWO COLUMNS WE WILL USE
pred_col = "prediction"
actual_col = 'target'

# CREATE THE CALIBRATION MODEL
ir = clb.generate_calibration_model(df_calibrate, pred_col, actual_col)
 
# APPLY THE CALIBRATION TO THE HOLDOUT
df_forecast['calibrated'] =  ir.transform( df_forecast[pred_col] )


df_final_forecast['calibrated'] =  ir.transform( df_final_forecast[pred_col] )

df_final_forecast['Calibrated Model'] = df_forecast['calibrated']

# PLOT THE ORIGINAL AND CALIBRATED PROBABILITIES
plt = clb.generate_reliability_plot(df_forecast, [pred_col, 'calibrated'], actual_col)
plt.savefig('results/calibration.png')

plt2 = clb.generate_reliability_plot(df_final_forecast, [ pred_col, 'Calibrated Model'], actual_col)
plt2.savefig('results/final_calibration.png')
 
#df_final_forecast.to_csv('results/final_holdout_calibrated.csv')

