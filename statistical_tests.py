import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import traceback


def load_data(csv_path):
    '''
    Load the data from the CSV file.
    '''

    # Check if the file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist.")
    
    # Load the data into a DataFrame
    df = pd.read_csv(csv_path)
    
    return df

def check_normality(data, column_name):
    '''
    Check if the data in the specified column is normally distributed.
    '''
    result = stats.anderson(data[column_name], dist='norm')
    print(f"Anderson-Darling Test Statistic: {result.statistic}")
    print(f"Critical Values: {result.critical_values}")
    print(f"Significance Levels: {result.significance_level}")
    if result.statistic < result.critical_values[2]:  # Compare with 5% significance level
        print(f"The data in column '{column_name}' appears to be normally distributed.")
        outcome = "normal"
    else:
        print(f"The data in column '{column_name}' does not appear to be normally distributed.")
        outcome = "not normal"
    return result.statistic, outcome

def plot_distribution(data, column_name, data_type):
    '''
    Plot the distribution of the data in the specified column.
    '''
    
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column_name], kde=True)
    plt.title(f'Distribution of {column_name}, for {data_type} data')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')

    # Save the plot to a file
    output_dir = "media/stats"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{column_name}_{data_type}_distribution.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory
    print(f"Plot saved to {plot_path}")

def perform_statistical_tests(dfs, column_names):
    ''' Perform statistical tests on two DataFrames.
     Mean and variance tests
    '''
    dist_results_table = []
    for column in column_names:
          for df in dfs:
            # Check if the column exists in the DataFrame
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame '{df.attrs['name']}'")
            else:
                stat,outcome = check_normality(df, column)
                # plot_distribution(df, column, df.attrs['name'])
                distribution_results = {
                    "Column": column,
                    "Data Type": df.attrs['name'],
                    "Anderson-Darling  Statistic": stat,
                    "Normality Outcome": outcome }
            
                dist_results_table.append(distribution_results)
            
       
    # Convert the results list to a DataFrame and print it
    dist_results_df = pd.DataFrame(dist_results_table)
    print("\nDistribution Test Results:")
    print(dist_results_df)

       
        
        
        # # T test for the means of two independent samples
        # t_stat, p_value = stats.ttest_ind(df1[column], df2[column])
        # # Check distribution of the data

        # # F test for the variances of two independent samples
        # f_stat, p_value_var = stats.f_oneway(df1[column], df2[column])    
        # # Print the results
        # print(f"T-test statistic: {t_stat}, p-value: {p_value}")
        # print(f"F-test statistic: {f_stat}, p-value: {p_value_var}")
        # # Create a dictionary to store the results
        # results = {
        #     "Column": column,
        #     "T-test Statistic": t_stat,
        #     "T-test p-value": p_value,
        #     "F-test Statistic": f_stat,
        #     "F-test p-value": p_value_var
        # }
        
        # # Append the results to a list
        # if 'results_table' not in locals():
        #     results_table = []
        # results_table.append(results)

        # # Convert the results list to a DataFrame and print it
        # results_df = pd.DataFrame(results_table)
        # print("\nStatistical Test Results:")
        # print(results_df)
        
from scipy.stats import shapiro


if __name__ == "__main__":
        try:
            # Load the data
            fault_data = load_data('./data/combined/fault_underhang_35g_bearing_500Hz.csv')
            fault_data.attrs['name'] = 'fault'
            normal_data = load_data('./data/combined/normal_500Hz.csv')
            normal_data.attrs['name'] = 'normal'
            column_names = [ 'tachometer_signal', 'underhang_bearing_radial', 'underhang_bearing_tangential','overhang_bearing_axial', 'overhang_bearing_radial', 'overhang_bearing_tangential', 'microphone']

            perform_statistical_tests([fault_data, normal_data], column_names)
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()  # Print the full traceback