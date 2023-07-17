# %%
__name__ == "__main__"

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from tqdm.auto import tqdm
from time import sleep
import scienceplots
import subprocess
pd.set_option('display.max_columns', None)
plt.style.use(['default','notebook']) #plt.style.use(['science','notebook'])
plt.tight_layout()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# %% [markdown]
# # Functions

# %%
columns = ['sur', 'sur2', 't', 'e1','e2','g1','g2','a1','i1','i2','i','spin1h','spintot',
            'beta','vp','spin1e','spin1q','spin2e','spin2q','spin2h','htot','m1','R1','m2',
            'R2','a2','m3','Roche1','R1_pow','type1','type2','type3','beta2','gamma','gamma2','flag'] #columns in output file
df_columns = ['N'] + columns #columns used in dataframes

output_directory0 = '/Users/bijan1339/Desktop/Research/Final_Output0/'
output_directory1 = '/Users/bijan1339/Desktop/Research/Final_Output1/'
output_directory6 = '/Users/bijan1339/Desktop/Research/Final_Output6/'
output_directory7 = '/Users/bijan1339/Desktop/Research/Final_Output/'
output_directory8 = '/Users/bijan1339/Desktop/Research/Final_Output8/'
output_directory9 = '/Users/bijan1339/Desktop/Research/Final_Output9/'

MS_Types = [0.,1.] #Main sequences keys from SSE
RG_Types = [3.,4.,5.,6.]# Giant Branch sequences keys from SSE
WD_Types = [10.,11.,12.] #White Dwarf keys from SSE



def get_first_line(file_path: str) -> str: #get first line 

    with open(file_path) as f:
        return f.readline()
#
def get_last_line(file_path: str) -> str: #faster way to get last line
    last_line=''
    with open(file_path, 'rb') as file:
        try:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
        except OSError:
            file.seek(0)

        last_line = file.readline().decode()
        return last_line

def is_within_percent(a,b, eps):
    '''returns whether a is within eps percentage of b
    :param eps: the proportion that you want a to be within b
    '''
    return abs(a/b - 1) <= eps
    
def cut_output_files(directory):
    #function to get final periods of all txt files in 'directory'
    #output in numpy array of logP in days

    for filename in os.listdir(directory):
        if filename.endswith('.txt') and filename.startswith('output'):
            this_file = os.path.join(directory, filename) #stores FULL filename

            first_line = get_first_line(this_file) #get first line
            last_line = get_last_line(this_file) #get last line
            
            N = filename.split('.txt')[0].split('_')[1] #get the output file number
            with open(directory+'cut_files/first_last_{}.txt'.format(N), 'w') as f:
                f.write(first_line)
                f.write(last_line)
def Roche_limit(q):
    '''
    Function to get Roche Limit of specified mass ratios (q)
    :param q: mass ratio
    :return: returns the Roche Limit (RHS of Eqn.1 from Naoz+2014)
    '''
    num1,num2=0.49,0.6
    return num1*np.power(q,2./3.)/(0.6* np.power(q,2./3.)+np.log(1+np.sqrt(q)))

directory = '/Users/bijan1339/Desktop/Research/OSPEStellarEv_updated/outputs/'
#get_final_periods(directory)  
def maybe_float(s):
    """
    Converts object type to float if it can be made a float
    :param s: The object (any type)
    :return: float(s) or s depending on whether it can be made float or not
    """
    try:
        return float(s)
    except (ValueError, TypeError):
        return s
def find_line(file,type1,type2=-1):
    """
    Goes through OSPE output file and finds last instance where there is an 
    inner binary with star types specified above (types from SSE).
    :param file: The first argument (string)
    :param type1: The second argument (integer)
    :param type2: The third argument. -1 means that the type can be any type (integer)
    :return: last line of occurence in file; line split into list and converted numbers to float (list)
    """
    if type2 == -1:
        for line in reversed(open(file).readlines()):
            line = [maybe_float(x) for x in line.split()]

            if (line[-6] == type1 or line[-7] == type1):
                return line
                break
    elif type1 == -1:
        for line in reversed(open(file).readlines()):
            line = [maybe_float(x) for x in line.split()]

            if (line[-6] == type2 or line[-6] == type2):
                return line
                break
                
    else: 
        for line in reversed(open(file).readlines()):
            line = [maybe_float(x) for x in line.split()]
            
            if ( (line[-6] == type1) and (line[-7] == type2) ) or ( (line[-6] == type2) and (line[-7] == type1) ): #order doesnt matter:
                return line
                break
def get_m1_condition(df):
    return ( (df['N'].astype(int)>1000) & (df['N'].astype(int)<=2000) ) #only returns 1m intial mass runs
def get_cassini_conditions(df,within=0.05):
    return ( 
        (df['Spin1P'] <= (1.+ within)*df['Omega_p1']) & (df['Spin1P'] >=(1. - within)*df['Omega_p1'])
            )

def powerlaw_distr(min : float, max : float, power : float) -> float:
    """Randomly generate value from a powerlaw distribution
    Args:
        min (float): minimum of powerlaw distribution
        max (float): minimum of powerlaw distribution
        power (float): exponent of powerlaw, P(x) ~ x^(power)

    Returns:
        float: randomly generated value from powerlaw distribution
    """
    
    powerp1 = power + 1.
    factor = ( (max / min)**powerp1 ) - 1.
    x = np.random.uniform(0,1)
    return min * (1. + factor*x) ** (1./powerp1)
def get_stability_crit(m1 : float, m2 : float, m3 : float, e2 : float, i : float) -> float:
    """_summary_

    Args:
        initial parameters

    Returns:
        float: returns (a2/a1)_crit, where if a2/a1 < (a2/a1)_crit, then unstable 
    """
    q_out = m3/(m1+m2)
    return 2.8 * np.power(1.+ q_out,2./5.) * np.power(1.+e2, 2./5.)*np.power(1.-e2, -6./5.)*(1-(0.3*i/180.))

k2 = 4*np.pi*np.pi

# %% [markdown]
# # Preparing Data

# %% [markdown]
# ## Concatenate Restart Files 

# %%
#GETTING RESTART FILENAMES
def concat_restart_files(output_directory : str, restart_directory : str):

    #Get list or restart output files
    restart_files = [x for x in os.listdir(restart_directory) if 'Re' in x]

    #concatting old with restarted files
    for this_restart_file in restart_files:

        n = int(this_restart_file.split('.')[0].split('_')[1])

        # Define the names of the two files to be concatenated
        file1 = directory + f'output_{n}.txt'
        file2 = restart_directory + this_restart_file

        # Open the files in read mode
        f1 = open(file1, "r")
        f2 = open(file2, "r")

        # Read the contents of the files
        contents1 = f1.read()
        contents2 = f2.read()

        # Concatenate the contents of the files
        concatenated_contents = contents1 + contents2

        # Define the name of the output file
        output_file = directory + f'output_{n}.txt'

        # Open the output file in write mode
        f_out = open(output_file, "w")

        # Write the concatenated contents to the output file
        f_out.write(concatenated_contents)

        # Close the files
        f1.close()
        f2.close()
        f_out.close()



# %%
directory = output_directory6
restart_directory = output_directory6 + 'Restart_Output/'
concat_restart_files(directory,restart_directory)

# %% [markdown]
# ## Fix the '\x00' Error

# %%
#Fix output files that dont have sur1, sur2, t in first line
def replace_first_line(myfile_path : str, line_to_replace : str):
    with open(myfile_path, "r") as f:
        lines = f.readlines()
        lines[0] = line_to_replace
    with open(myfile_path, "w") as f:
        f.writelines(lines)

def fix_bin_files(output_directory = '/Users/bijan1339/Desktop/Research/Final_Output8/Restart_Output/',
                  og_directory = '/Users/bijan1339/Desktop/Research/Final_Output8/'):
    for filename in os.listdir(output_directory):
        
        this_file = os.path.join(output_directory, filename) #stores FULL filename
        n = filename.split('_')[-1].split('.')[0]
        og_file_path = og_directory + f'output_{n}.txt'
        
        second_line=''
        last_og_line=''
        first_line = get_first_line(this_file).replace('\x00','').split('\t')
        
        first_line = [x.strip() for x in first_line]
        
        if len(first_line) == 33: #length should be 36 with sur1,sur2,sur3
            sur1,sur2,t = '2','2','0.0'
            #change the above params using context of the file
            #if os.stat(this_file).st_size > 7e3: #if the file has more that one line
            with open(og_file_path, "r") as f:
                lines = f.readlines()
                second_last_line = lines[-2].replace('\x00','').split('\t') if len(lines) >= 2 else ''
                
                if len(second_line) > 1: 
                    second_line = [x.strip() for x in second_line]
                    sur1,sur2,t = second_line[0],second_line[1],second_line[2]
                else: #if there is only one line, go to the original output.txt file and get time from last
                    # line if it has len() > 34, else just get time from second to last line
                    last_og_line = get_last_line(og_file_path).replace('\x00','').split('\t') #check the last printout of the normal output file
                    if len(last_og_line) > 34:
                        sur1,sur2,t = last_og_line[0],last_og_line[1], last_og_line[2]
                    elif len(second_last_line) > 34:
                        sur1,sur2,t = second_last_line[0], second_last_line[1], second_last_line[2]
                    
                
                mod_first_line = '\t'.join(   [sur1, sur2, t] + first_line   ) + '\n'
            
            replace_first_line(this_file, mod_first_line)
        
###############################################
###########   MAKING DATAFRAME(s)  ############
###############################################

#FUNCTIONS TO ADD COLUMNS TO END OF DATAFRAME
def get_inner_period_row(df: pd.DataFrame):
    '''returns VECTORIZED period column in DAYS from df '''
    return np.sqrt((df['a1']**3) / (df['m1'] + df['m2']) )*365.25
def get_outer_period_row(df: pd.DataFrame):
    '''returns VECTORIZED period column in DAYS from df '''
    return np.sqrt((df['a2']**3) / (df['m1'] + df['m2'] + df['m3']) )*365.25
def get_logg(df: pd.DataFrame, G = k2):
    '''returns log(g) for m1,m2 from df row'''
    g1 = G*df['m1']/(df['R1']*df['R1'])
    g2 = G*df['m2']/(df['R2']*df['R2'])
    return [np.log10(g1), np.log10(g2)]
def get_Omega_p(df: pd.DataFrame ):
    beta = df['beta']*np.pi/180.
    beta2 = df['beta2']*np.pi/180.
    omega_1 = (4*np.pi/df['P_in'] ) / (np.cos(beta) + (1/np.cos(beta)) )   
    omega_2 = (4*np.pi/df['P_in'] ) / (np.cos(beta2) + (1/np.cos(beta2)) )   
    return [2*np.pi/omega_1,2*np.pi/omega_2] #returns expected spin periods in days

def get_spin_periods(df: pd.DataFrame):
    
    spin1P_radyr = np.sqrt(df['spin1h']**2.+df['spin1e']**2.+df['spin1q']**2.)
    spin2P_radyr = np.sqrt(df['spin2h']**2.+df['spin2e']**2.+df['spin2q']**2.)
    spin1P = 2.*np.pi*365.25/spin1P_radyr
    spin2P = 2.*np.pi*365.25/spin2P_radyr
    return [spin1P,spin2P]

def get_star_type(row,type_col = 'type1'): #type_col =  'type1' or ;type2' or 'type3'
    MS_Types = [0.,1.] #Main sequences keys from SSE
    RG_Types = [3.,4.,5.,6.]# Giant Branch sequences keys from SSE
    WD_Types = [10.,11.,12.] #White Dwarf keys from SSE
    if row[type_col] in MS_Types: return 'MS'
    if row[type_col] in RG_Types: return 'RG'
    if row[type_col] in WD_Types: return 'WD'
    else: return 'other'

def get_bin_type(row): 
    if row['startype1'] == 'MS' and row['startype2'] == 'MS': return 'MSMS'
    if row['startype1'] == 'RG' and row['startype2'] == 'RG': return 'RGRG'
    if row['startype1'] == 'WD' and row['startype2'] == 'WD': return 'WDWD'
    
    if (row['startype1'] == 'MS' and row['startype2'] == 'RG') or (row['startype2'] == 'MS' and row['startype1'] == 'RG'):  return 'RGMS'
    if (row['startype1'] == 'MS' and row['startype2'] == 'WD') or (row['startype2'] == 'MS' and row['startype1'] == 'WD'):  return 'WDMS'
    if (row['startype1'] == 'RG' and row['startype2'] == 'WD') or (row['startype2'] == 'RG' and row['startype1'] == 'WD'):  return 'RGWD'
    
    else: return 'other'

def get_trip_type(row): 
    if row['bintype'] == 'MSMS':
        if row['startype3'] == 'MS': return 'MSMS-MS'
        if row['startype3'] == 'RG': return 'MSMS-RG'
        if row['startype3'] == 'WD': return 'MSMS-WD'
        else: return 'MSMS-XX'
    if row['bintype'] == 'RGRG':
        if row['startype3'] == 'MS': return 'RGRG-MS'
        if row['startype3'] == 'RG': return 'RGRG-RG'
        if row['startype3'] == 'WD': return 'RGRG-WD'
        else: return 'RGRG-XX'
        
    if row['bintype'] == 'WDWD':
        if row['startype3'] == 'MS': return 'WDWD-MS'
        if row['startype3'] == 'RG': return 'WDWD-RG'
        if row['startype3'] == 'WD': return 'WDWD-WD'
        else: return 'WDWD-XX'
        
    if row['bintype'] == 'RGMS':
        if row['startype3'] == 'MS': return 'RGMS-MS'
        if row['startype3'] == 'RG': return 'RGMS-RG'
        if row['startype3'] == 'WD': return 'RGMS-WD'
        else: return 'RGMS-XX'
    if row['bintype'] == 'WDMS':
        if row['startype3'] == 'MS': return 'WDMS-MS'
        if row['startype3'] == 'RG': return 'WDMS-RG'
        if row['startype3'] == 'WD': return 'WDMS-WD'
        else: return 'WDMS-XX'
    if row['bintype'] == 'RGWD':
        if row['startype3'] == 'MS': return 'RGWD-MS'
        if row['startype3'] == 'RG': return 'RGWD-RG'
        if row['startype3'] == 'WD': return 'RGWD-WD'
        else: return 'RGWD-XX'
    else: return 'other'

def make_dataframes(directory : str = '/Users/bijan1339/Desktop/Research/Final_Output/',
                    start : int = 8000, stop : int =9000) -> tuple:
    """
    Generates appends initial and final lines of output files to data frames

    Args:
        directory (str): directory with output files. 
        start (int): minimum n (output_n.txt)
        stop (int): maximum n 

    Returns:
        initial, final (tuple): initial, final contain first, last line dataframes without any conditions
    """    
    
    #columns in output file
    
    initial = pd.DataFrame(columns=df_columns) #initialize dfs
    final = pd.DataFrame(columns=df_columns)

    not_finished=[]
    didnt_start=[]
    for filename in os.listdir(directory):
        this_file = os.path.join(directory, filename) #stores FULL filename
        
        n = filename.split('_')[-1].split('.')[0]

        if (  filename.endswith('.txt') and (filename.startswith('output_') or filename.startswith('concatenated_')) and 
            ( (start <= int(n) <= stop) ) and os.stat(this_file).st_size != 0.
        ):
            #n = filename.split('_')[-1].split('.')[0]

            first_line = get_first_line(this_file)#.replace('\x00','') #get first line
            last_line = get_last_line(this_file)#.replace('\x00','') #get last line 

            first_line, last_line = first_line.split('\t'), last_line.split('\t')
            first_line, last_line  = [x.strip() for x in first_line],[x.strip() for x in last_line] #remove white spaces (' ' and '\n')
            #first_line, last_line  = [float(x) for x in first_line[:-1]],[float(x) for x in first_line[:-1]]
            first_line.insert(0,n),last_line.insert(0,n)

            """
            a = float(last_line[df_columns.index('a1')]) #store final SMA (in AU)
            a2 = float(last_line[df_columns.index('a2')])
            e = float(last_line[df_columns.index('e1')]) #store final e (in AU)
            e2 = float(last_line[df_columns.index('e2')]) #store final e (in AU)
            R1,R2 = float(last_line[df_columns.index('R1')]), float(last_line[columns.index('R2')])
            m1,m2 = float(last_line[df_columns.index('m1')]), float(last_line[columns.index('m2')])
            t = float(last_line[df_columns.index('t')])
            P = np.sqrt(a**3 / (m1+m2) )*365.25
            
            Roche1=Roche_limit(m1/m2)
            Roche2=Roche_limit(m2/m1)
            Roche1_criteria = R1 > (a*(1-e)*Roche1)
            Roche2_criteria = R2 > (a*(1-e)*Roche2)
            tidal_locking = (e <= 5e-5 and P*365.25<=7.) #and float(last_line[columns.index('sur')])== 4. #tidal locking if sur==4
            time_criteria = t >= 1e10 # check if until time completion
            epsilon_criteria = (a*e2)/(a2*(1-e2**2)) > 0.1 # essentially stability criteria
            major_criteria = first_line==last_line
            
            #end_criteria = last_line[-1] =='END'
            unfinished = not time_criteria and not (Roche1_criteria or Roche2_criteria or epsilon_criteria or tidal_locking)
            finished =  last_line[-1] == 'END' or float(last_line[3]) > 10e9
            """
            initial.loc[len(initial)]=first_line
            final.loc[len(final)]=last_line

    #default columns to float
    for i in range(1,len(final.columns)-1):
        final.iloc[:, i] = final.iloc[:, i].astype(float)
        initial.iloc[:, i] = initial.iloc[:, i].astype(float)
    

    #Correcting Dtypes
    initial['N'], final['N'] = initial['N'].astype(int), final['N'].astype(int)
    initial['flag'], final['flag'] = initial['flag'].astype(str), final['flag'].astype(str)
    final['sur'], final['sur2'] = final['sur'].astype(int), final['sur2'].astype(int)
    initial['sur'], initial['sur2'] = initial['sur'].astype(int), initial['sur2'].astype(int)
    final['type1'], final['type2'], final['type3'] = final['type1'].astype(int), final['type2'].astype(int), final['type3'].astype(int)
    initial['type1'], initial['type2'], initial['type3'] = initial['type1'].astype(int), initial['type2'].astype(int), initial['type3'].astype(int)

    #ADDING USEFUL COLUMNS AT END OF DFs
    final['P_in'], final['P_out'] = get_inner_period_row(final), get_outer_period_row(final)
    initial['P_in'], initial['P_out'] = get_inner_period_row(initial), get_outer_period_row(initial)

    # initial['m1/m2'], final['m1/m2'] = initial['m1'] / initial['m2'], final['m1'] / final['m2']
    
    # final['logg_m1'], final['logg_m2'] = get_logg(final)[0], get_logg(final)[1]
    # initial['logg_m1'], initial['logg_m2'] = get_logg(initial)[0], get_logg(initial)[1]

    # final['Omega_p1'], final['Omega_p2'] = get_Omega_p(final)[0], get_Omega_p(final)[1]
    # final['Spin1P'], final['Spin2P'] = get_spin_periods(final)[0], get_spin_periods(final)[1]

    # initial['Omega_p1'], initial['Omega_p2'] = get_Omega_p(initial)[0], get_Omega_p(initial)[1]
    # initial['Spin1P'], initial['Spin2P'] = get_spin_periods(initial)[0], get_spin_periods(initial)[1]
        
    final['startype1'] = final.apply(lambda row: get_star_type(row,'type1'), axis=1)
    final['startype2'] = final.apply(lambda row: get_star_type(row,'type2'), axis=1)
    final['startype3'] = final.apply(lambda row: get_star_type(row,'type3'), axis=1)
    initial['startype1'] = initial.apply(lambda row: get_star_type(row,'type1'), axis=1)
    initial['startype2'] = initial.apply(lambda row: get_star_type(row,'type2'), axis=1)
    initial['startype3'] = initial.apply(lambda row: get_star_type(row,'type3'), axis=1)

    final['bintype'] = final.apply(lambda row: get_bin_type(row), axis=1)
    final['triptype'] = final.apply(lambda row: get_trip_type(row), axis=1)
    initial['bintype'] = initial.apply(lambda row: get_bin_type(row), axis=1)
    initial['triptype'] = initial.apply(lambda row: get_trip_type(row), axis=1)

    initial = initial[initial['N'].isin(final['N'])]
    return initial, final

# %%
# %% [markdown]
# ## OutFile --> DF

# %%
def make_output_df(output_file : str = '/Users/bijan1339/Desktop/Research/Final_Output/output_7001.txt'):
    
    columns = ['sur', 'sur2', 't', 'e1','e2','g1','g2','a1','i1','i2','i','spin1h','spintot',
                'beta','vp','spin1e','spin1q','spin2e','spin2q','spin2h','htot','m1','R1','m2',
                'R2','a2','m3','Roche1','R1_pow','type1','type2','type3','beta2','gamma','gamma2','flag']
    all_binary_types = pd.DataFrame(columns= df_columns + ['binary_type']) #initialize df, include last binary_type column
    
    filename = output_file.split('/')[-1]
    n = maybe_float(  filename.split('_')[-1].split('.')[0]  ) #output_n.txt

    if filename.startswith('output'):
        try:
            this_df = pd.read_csv(output_file, delimiter='\t', header=None)
            this_df.columns = columns
            
            return this_df

        except pd.errors.EmptyDataError: #if file is empty
            return None
    else: 
        print("Invalid Input FileName")
        return None
                        
def create_dfs(initial: pd.DataFrame, final: pd.DataFrame):
  
  df_final = final.query("(flag == 'END') or (t >= 1e10)")
  df_initial = initial[initial['N'].isin(df_final['N'])]

  unf_final = final.query("(flag != 'END') and (t != 0.)")   
  unf_initial = initial[initial['N'].isin(unf_final['N'])]


  close_df_final = df_final.query("(P_in <= 16) and (sur != 0.) and (t != 0)")
  close_df_initial = df_initial[df_initial['N'].isin(close_df_final['N'])] 

  merged_df_final = df_final[(df_final['sur']==0)]
  merged_df_initial = df_initial[df_initial['N'].isin(merged_df_final['N'])]

  return df_initial, df_final, unf_initial, unf_final, close_df_initial, close_df_final, merged_df_initial, merged_df_final

#convert all period column to same units in "Period" columns
def all_to_day(row):
    '''returns period in DAYS from df row'''
    if row['x_Per'] == 'y': return row['Per'] * 365.25 #yr to days
    if row['x_Per'] == 'k': return row['Per'] * 1000. * 365.25 #kyr to days
    if row['x_Per'] == 'd': return row['Per'] #alr days
def add_out_params(row,df):
    ID = row['WDS']
    this_outer_row = df[df['WDS'] == ID]
    P_out, e_out, a_out = this_outer_row['P'],this_outer_row['e'],this_outer_row['a']
    try: 
        return (P_out.iloc[0], e_out.iloc[0], a_out.iloc[0]) 
    except IndexError: 
        return [0.0,0.0,0.0]

def get_combin_values(final):
    combinations = final.groupby(['triptype']).size().reset_index().rename(columns={0:'Sum'})
    bin_types = ['MSMS', 'RGMS', 'RGRG', 'RGWD', 'WDMS', 'WDWD', 'other']
    bin_types = ['MSMS', 'RGMS', 'RGWD', 'WDMS', 'WDWD']

    third_types = ['MS','RG','WD','XX']

    vals = np.array([])
    for t in bin_types:
        XXXX_MS = 0
        XXXX_RG = 0
        XXXX_WD = 0
        XXXX_other = 0
        this_comb = combinations[combinations.triptype.str.contains(t)]
        for third_star in third_types:
            this_trip = this_comb[this_comb.triptype.str.contains(f'-{third_star}')] #eg, this_trip is 'MSMS-MS

            this_val = this_trip.values[0,1] if this_trip.size != 0 else 0 #this_val is the sum of triples with certain combination
            vals = np.append(vals,this_val) 
            
    N = 4    
    vals = np.array(  [vals[n:n+N] for n in range(0, len(vals), N)]   ) #put in groups of 4

    return vals

def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target.get_shared_x_axes().join(target, ax)
        if sharey:
            target.get_shared_y_axes().join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


# %%
def tight_pairs(n_cols, fig=None):
    """
    Stitch vertical pairs together.

    Input:
    - n_cols: number of columns in the figure
    - fig: figure to be modified. If None, the current figure is used.

    Assumptions: 
    - fig.axes should be ordered top to bottom (ascending row number). 
      So make sure the subplots have been added in this order. 
    - The upper-half's first subplot (column 0) should always be present

    Effect:
    - The spacing between vertical pairs is reduced to zero by moving all lower-half subplots up.

    Returns:
    - Modified fig
    """
    if fig is None:
        fig = plt.gcf()
    for ax in fig.axes:
        if hasattr(ax, 'get_subplotspec'):
            ss = ax.get_subplotspec()
            row, col = ss.num1 // n_cols, ss.num1 % n_cols
            if (row % 2 == 0) and (col == 0): # upper-half row (first subplot)
                y0_upper = ss.get_position(fig).y0
            elif (row % 2 == 1): # lower-half row (all subplots)
                x0_low, _ , width_low, height_low = ss.get_position(fig).bounds
                ax.set_position(pos=[x0_low, y0_upper - height_low, width_low, height_low])
    return fig

def epsilon(a1,a2,e2):
    return (   (a1/a2)* (  e2/(1-e2**2)  )  )

def L1(m1,m2,a,e):
    k2 = 4*np.pi*np.pi
    mu = m1*m2 / (m1+m2)
    M = m1+m2
    return mu * np.sqrt( k2*M*a*(1-e*e)  )

def e_const_L(L,m1,m2,a):
    """
    returns e for Constant Angular Momentum
    """
    k2 = 4*np.pi*np.pi
    mu = m1*m2 / (m1+m2)
    M = m1+m2
    
    return np.sqrt( 1 - ( L**2 / (mu*mu*k2*M*a) )   )

## Getting initial dfs
def get_initial_N_a2(initial_df,input_df,max_a2):
    """
    Gets the N's (IDs) in input_df for those that initially had an a2 < a2_max
    """
    return initial_df[(initial_df.N.isin(input_df.N)) & (initial_df.a2 < max_a2)].N.values

def get_tidal_locked_condition(row):
    a1,Roche1,e1,P1 = row['a1'],row['Roche1'],row['e1'],row['P_in']
    beta,spintot,spin2h = row['beta'],row['spintot'],row['spin2h']
    M_PI = np.pi
    
    tl_crit1 = (a1/Roche1 <= 5 or a1 <= 0.1) 
    tl_crit2 = ( e1 <= 1e-4 )
    tl_crit3 = (a1 <= 0.1 and e1 <= 1e-2)

    tl_condition = ( e1 <= 1e-2 and tl_crit1 and tl_crit2) or tl_crit3
    return tl_condition

def get_t_ekl(m1,m2,m3,e2,P1,P2):
    """
    Characteristic quadrupole timescale fro EKL (eq 27 from review)
    """
    return (8./(15*np.pi))*((m1+m2+m3)/m3)*(P2*P2/P1)*np.sqrt(1-e2*e2)*(1-e2*e2)


# %%
def add_tekl_col(df):
    df.loc[:,'t_quad'] = get_t_ekl(df.m1,df.m2,df.m3,df.e2,df.P_in, df.P_out)
    df.loc[:,'t_remain'] = 12.5e9 - df.t
    df.loc[:,'epsilon'] = (df.a1/df.a2) * df.e2 / (1-df.e2**2)
    df.loc[:,'stability'] = get_stability_crit(df.m1,df.m2,df.m3,df.e2,df.i)
    df.loc[:,'a2/a1'] = df.a2/df.a1



