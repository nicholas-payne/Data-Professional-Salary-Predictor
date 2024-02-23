import numpy as np
import pandas as pd
import flask
import pickle

app = flask.Flask(__name__,template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        #converting data from flask to one-row dataframe, filling in blanks where needed to allow for imputation
        input_items = ['work_year','experience_level','employment_type',
                       'job_title','employee_residence','remote_ratio',
                       'company_location','company_size']
        imputed_values = [2023, 'SE', 'FT', 'Data Engineer', 'US', 0, 'US', 'M']
        
        input_list = []
        for idx,item in enumerate(input_items):
            if flask.request.form[item]=="":
                                    input_list.append(imputed_values[idx])
            else:
                input_list.append(flask.request.form[item])
        
        df = pd.DataFrame(input_list).transpose()
        df.columns = input_items
        
        #Preparing data to fit to model
        with open('model/replacements.pkl','rb') as file:
            replacements = pickle.load(file)
               
        df['job_title_clean'] = df['job_title'].map(replacements)
        
        country_gdp_coor = pd.read_csv('model/GDP_and_coordinates.csv',
                                       index_col='country_code')
        
        df = df.merge(country_gdp_coor,
                      left_on='employee_residence',
                      right_index=True,
                      how='left')
        df.rename(columns={'GDP':'employee_residence_gdp',
                           'latitude':'employee_latitude',
                           'longitude':'employee_longitude'},
                  inplace=True)

        df = df.merge(country_gdp_coor,
                      left_on='company_location',
                      right_index=True,
                      how='left')
        df.rename(columns={'GDP':'company_location_gdp',
                           'latitude':'company_latitude',
                           'longitude':'company_longitude'},
                  inplace=True)
        
        for item in ['employee_','company_']:
            df[f'{item}x'] = np.cos(df[f'{item}latitude']) * np.cos(df[f'{item}longitude'])
            df[f'{item}y'] = np.cos(df[f'{item}latitude']) * np.sin(df[f'{item}longitude'])
            df[f'{item}z'] = np.sin(df[f'{item}latitude'])
            
        df['years_since_ref'] = df['work_year'].astype(int) - 2020
        df['remote_ratio'] = df['remote_ratio'].astype(int) / 100

        df['outsourced'] = df['employee_residence'] != df['company_location']
        df['outsourced'] = df['outsourced'].astype('int')
        
        #Creating model input object and dropping any remaining unused columns
        model_input = df.drop(['job_title','employee_residence','company_location',
                               'employee_latitude','employee_longitude','company_latitude',
                               'company_longitude','work_year'],
                              axis=1)
        #Fitting the Model
        with open('model/salary_predictor_median.pkl', 'rb') as file:
            model_median = pickle.load(file)

        with open('model/salary_predictor_90.pkl', 'rb') as file:
            model_90 = pickle.load(file)

        with open('model/salary_predictor_10.pkl', 'rb') as file:
            model_10 = pickle.load(file)
        
        quantile_50 = model_median.predict(model_input)
        quantile_90 = model_90.predict(model_input)
        quantile_10 = model_10.predict(model_input)
                
        return(flask.render_template('main.html',
                                     original_input = flask.request.form,
                                     median = "${:0,}".format(int(10**quantile_50[0])),
                                     perc90 = "${:0,}".format(int(10**quantile_90[0])),
                                     perc10 = "${:0,}".format(int(10**quantile_10[0]))))

if __name__ == "__main__":
    app.run()