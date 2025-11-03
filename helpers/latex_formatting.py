import pandas as pd

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
    df['Coefficient'] = df.apply(
        lambda row: f"\\makecell[tr]{{{sig_coef(row)} \\\\ ({row['stderror']:.3f})}}", axis=1)
    df = df[['Coefficient']]
    df.loc['R-squared'] = [f"{results.rsquared:.3f}"]
    df.loc['Observations'] = [f"{int(results.nobs)}"]
    return df

# table with one regression - no concatenating
def export_single_regression(df, caption, label, widthmultiplier = 1.0):
    df = df.rename({
        'np.log(rent)': '(log) Rent',
        'np.log(valueh)': '(log) Home Value',
        'hwy': 'Highway',
        'mblack_1945def': 'Majority Black (60\\% Threshold)',
        'mblack_1945def:Residential': 'Majority Black (60\\% Threshold) x Residential',
        'mblack_mean_pct': 'Majority Black (Avg. Percent)',
        'mblack_mean_pct:Residential': 'Majority Black (Avg. Percent) x Residential',
        'mblack_mean_share': 'Majority Black (Avg. Share)',
        'mblack_mean_share:Residential': 'Majority Black (Avg. Share) x Residential',
        'distance_to_cbd': 'Distance to CBD',
        'dist_to_hwy': 'Distance to Nearest Highway (1940)'}, axis = 'index')
    
    # format for latex output
    num_cols = df.shape[1]
    col_format = '@{\\extracolsep{\\fill}}l*' + f'{{{num_cols}}}' + '{r}'
    text = df.style.format(precision=2).to_latex(position_float = 'centering',
                caption=caption, position = 'h', label=label, hrules=True, column_format = col_format)
    text = text.replace('\\begin{tabular}', f'\\begin{{tabular*}}{{{widthmultiplier}\\textwidth}}').replace('\\end{tabular}', '\\end{tabular*}')
    filename = label.split(':')[-1] + '.tex'
    with open('tables/' + filename, 'w') as f:
        f.write(text)

# table with multiple regressions - definition of 'Black' as column title
def export_multiple_regressions(df_list, caption, label):
    def column_names(df):
        if 'mblack_1945def' in df.index:
            return df.rename(columns = {'Coefficient':'60\\% Threshold'})
        elif 'mblack_mean_pct' in df.index:
            return df.rename(columns = {'Coefficient':'Avg. Percent'})
        elif 'mblack_mean_share' in df.index:
            return df.rename(columns = {'Coefficient':'Avg. Share'})
        else:
            return df
    def standardize_index(df):
        return df.rename({
        'np.log(rent)': '(log) Rent',
        'np.log(valueh)': '(log) Home Value',
        'hwy': 'Highway',
        'mblack_1945def': 'Black',
        'mblack_1945def:Residential': 'Black x Residential',
        'mblack_mean_pct': 'Black',
        'mblack_mean_pct:Residential': 'Black x Residential',
        'mblack_mean_share': 'Black',
        'mblack_mean_share:Residential': 'Black x Residential',
        'distance_to_cbd': 'Distance to CBD',
        'dist_to_hwy': 'Distance to Nearest Highway (1940)'}, axis = 'index')
    
    renamed_list = [standardize_index(column_names(df)) for df in df_list]    
    df = pd.concat(renamed_list, axis = 1)
    
    # format for latex output
    num_cols = df.shape[1]
    col_format = '@{\\extracolsep{\\fill}}l*' + f'{{{num_cols}}}' + '{r}'
    text = df.style.format(precision=2).to_latex(position_float = 'centering',
                caption=caption, position = 'h', label=label, hrules=True, column_format = col_format)
    text = text.replace('\\begin{tabular}', f'\\begin{{tabular*}}{{\\textwidth}}').replace('\\end{tabular}', '\\end{tabular*}')
    filename = label.split(':')[-1] + '.tex'
    with open('tables/' + filename, 'w') as f:
        f.write(text)