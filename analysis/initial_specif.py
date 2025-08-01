import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

# function to export df as latex table with full page width and add'l formatting
def format_regression_results(results):
    df = pd.DataFrame({'coef':results.params, 'stderror': results.bse, 'pvalue': results.pvalues})[1:]
    df['pvalue'] = results.pvalues[1:]
    def sig_coef(row):
        if row['pvalue'] < 0.001:
            return f"{row['coef']:.3f}^{{***}}"
        elif row['pvalue'] < 0.01:
            return f"{row['coef']:.3f}^{{**}}"
        elif row['pvalue'] < 0.05:
            return f"{row['coef']:.3f}^{{*}}"
        else:
            return f"{row['coef']:.3f}"
    df['Coefficient'] = df.apply(lambda row: f"{sig_coef(row)}\n({row['stderror']:.3f})", axis=1)
    df = df[['Coefficient']]
    df.index.name = 'Variable'
    df.loc['R-squared'] = [f"{results.rsquared:.3f}"]
    df.loc['Observations'] = [f"{int(results.nobs)}"]
    return df

# table with one regression - no concatenating
def export_single_regression(df, caption, label, widthmultiplier = 1.0):
    df = df.rename({
        'rent': 'Rent',
        'valueh': 'Home Value',
        'hwy': 'Highway',
        'mblack_1945def': 'Majority Black (60\% Threshold)',
        'mblack_1945def:Residential': 'Majority Black (60\% Threshold) x Residential',
        'mblack_mean_pct': 'Majority Black (Avg. Percent)',
        'mblack_mean_pct:Residential': 'Majority Black (Avg. Percent) x Residential',
        'mblack_mean_share': 'Majority Black (Avg. Share)',
        'mblack_mean_share:Residential': 'Majority Black (Avg. Share) x Residential',
        'distance_to_cbd': 'Distance to CBD'}, axis = 'index')
    
    # format for latex output
    num_cols = df.shape[1]
    col_format = '|'.join(['X'] * (num_cols  + 1))
    text = df.style.to_latex(position_float = 'centering',
                caption=caption, position = 'h', label=label, hrules=True)
    text = text.replace('\\begin{tabular}', f'\\begin{{tabularx}}{{{widthmultiplier}\\textwidth}}{{{col_format}}}').replace('\\end{tabular}', '\\end{tabularx}')
    filename = label.split(':')[-1] + '.tex'
    with open('tables/' + filename, 'w') as f:
        f.write(text)

# table with multiple regressions - definition of 'Black' as column title
def export_multiple_regressions(df_list, caption, label):
    df = pd.concat(df_list, axis = 1)
    def column_names(df):
        if 'mblack_1945def' in df.index:
            return df.rename(columns = {'Coefficient':'(60\% Threshold)'})
        elif 'mblack_mean_pct' in df.index:
            return df.rename(columns = {'Coefficient':'(Avg. Percent)'})
        elif 'mblack_mean_share' in df.index:
            return df.rename(columns = {'Coefficient':'(Avg. Share)'})
        else:
            return df
    df = column_names(df).apply()
    df = df.rename({
        'rent': 'Rent',
        'valueh': 'Home Value',
        'hwy': 'Highway',
        'mblack_1945def': 'Black',
        'mblack_1945def:Residential': 'Black x Residential',
        'mblack_mean_pct': 'Black',
        'mblack_mean_pct:Residential': 'Black x Residential',
        'mblack_mean_share': 'Black',
        'mblack_mean_share:Residential': 'Black x Residential',
        'distance_to_cbd': 'Distance to CBD'}, axis = 'index')
    
    # format for latex output
    num_cols = df.shape[1]
    col_format = '|'.join(['X'] * (num_cols  + 1))
    text = df.style.to_latex(position_float = 'centering',
                caption=caption, position = 'h', label=label, hrules=True)
    text = text.replace('\\begin{tabular}', f'\\begin{{tabularx}}{{\\textwidth}}{{{col_format}}}').replace('\\end{tabular}', '\\end{tabularx}')
    filename = label.split(':')[-1] + '.tex'
    with open('tables/' + filename, 'w') as f:
        f.write(text)


atl_sample = pd.read_pickle('data/output/atl_sample.pkl')

model_naive = 'hwy ~ mblack_1945def + Residential + np.log(rent) + np.log(valueh) + distance_to_cbd'
results_naive = format_regression_results(smf.ols(model_naive, data=atl_sample).fit(cov_type='HC3'))

model_1945def = 'hwy ~ mblack_1945def + Residential + (mblack_1945def * Residential) + np.log(rent) + np.log(valueh) + distance_to_cbd'
results_1945def = format_regression_results(smf.ols(model_1945def, data=atl_sample).fit(cov_type='HC3'))

model_pct = 'hwy ~ mblack_mean_pct + Residential + (mblack_mean_pct * Residential) + np.log(rent) + np.log(valueh) + distance_to_cbd'
results_pct = format_regression_results(smf.ols(model_pct, data=atl_sample).fit(cov_type='HC3'))

model_share = 'hwy ~ mblack_mean_share + Residential + (mblack_mean_share * Residential) + np.log(rent) + np.log(valueh) + distance_to_cbd'
results_share = format_regression_results(smf.ols(model_share, data=atl_sample).fit(cov_type='HC3'))

# naive regression in its own table
export_single_regression(results_naive, caption = 'Naive Regression Results', label = 'tab:naive_results', widthmultiplier=0.7)

# other results together ?
export_multiple_regressions([results_1945def, results_pct, results_share],
                            caption = 'Determinants of Highway Placement',
                            label = 'tab:initial_results')
