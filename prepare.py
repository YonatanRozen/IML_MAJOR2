import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def prepare_data(training_data, new_data):

  # Copy the data
  train_df = training_data.copy()
  new_df = new_data.copy()

  # Add the SpecialProperty column & drop blood_type
  group_True = ['O+', 'B+']
  train_df['SpecialProperty'] = np.where(train_df['blood_type'].isin(group_True), 1, -1)
  new_df['SpecialProperty'] = np.where(new_df['blood_type'].isin(group_True), 1, -1)
  train_df = train_df.drop('blood_type', axis=1)
  new_df = new_df.drop('blood_type', axis=1)

  # Split the symptoms variable into columns
  train_df_new_cols = train_df['symptoms'].str.get_dummies(';')
  train_df_new_cols[train_df_new_cols == 0] = -1

  new_df_new_cols = new_df['symptoms'].str.get_dummies(';')
  new_df_new_cols[new_df_new_cols == 0] = -1

  train_df = pd.concat([train_df, train_df_new_cols], axis=1)
  new_df = pd.concat([new_df, new_df_new_cols], axis=1)

  train_df = train_df.drop('symptoms', axis=1)
  new_df = new_df.drop('symptoms', axis=1)

  # Split the gender column into "Male" and "Female"
  train_df['Female'] = np.where(train_df['sex'] == 'F', 1, -1)
  train_df['Male'] = np.where(train_df['sex'] == 'M', 1, -1)
  train_df = train_df.drop('sex', axis=1)

  new_df['Female'] = np.where(new_df['sex'] == 'F', 1, -1)
  new_df['Male'] = np.where(new_df['sex'] == 'M', 1, -1)
  new_df = new_df.drop('sex', axis=1)

  # Split location column into location_x and location_y
  train_df['location_x'] = [float(location.split("'")[1]) for location in train_df['current_location']]
  train_df['location_y'] = [float(location.split("'")[3]) for location in train_df['current_location']]
  train_df = train_df.drop('current_location', axis=1)

  new_df['location_x'] = [float(location.split("'")[1]) for location in new_df['current_location']]
  new_df['location_y'] = [float(location.split("'")[3]) for location in new_df['current_location']]
  new_df = new_df.drop('current_location', axis=1)

  train_df = train_df.drop('pcr_date', axis=1)
  train_df = train_df.drop('patient_id', axis=1)
  new_df = new_df.drop('pcr_date', axis=1)
  new_df = new_df.drop('patient_id', axis=1)


  # Columns designated for minmax scaling
  cols_for_minmax = ['PCR_01','PCR_02','PCR_03','PCR_06','cough','fever',
                    'shortness_of_breath','smell_loss','sore_throat','Female','Male', 'SpecialProperty']
  # Columns designated for standrad scaling
  cols_for_standard = ['age','weight','num_of_siblings','happiness_score','household_income',
                   'conversations_per_day','sugar_levels','sport_activity','location_x',
                   'location_y','PCR_04','PCR_05','PCR_07','PCR_08','PCR_09','PCR_10']

  # minmax scale all the minmax columns (in both training & test data, fit only training)
  scaler = MinMaxScaler(feature_range=(-1,1))
  scaler.fit(train_df[cols_for_minmax])
  new_df[cols_for_minmax] = scaler.transform(new_df[cols_for_minmax])

  # Standard scale all the standard columns
  scaler = StandardScaler()
  scaler.fit(train_df[cols_for_standard])
  new_df[cols_for_standard] = scaler.transform(new_df[cols_for_standard])

  return new_df