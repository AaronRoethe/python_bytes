
from zipfile import ZipFile
from pathlib import Path

def extract_files(start, extract_extention, end):
    # z = Path(f'data/load')
    z = Path(f'{start}')
    b = list(z.glob('*.zip'))
    for i in b:
        with ZipFile(i, 'r') as zip:
            listOfFileNames = zip.namelist()
            for fileName in listOfFileNames:
                if fileName.endswith(f'.{extract_extention}'):
                    zip.extractall(f'{end}')

extract_files('data/load', 'csv', 'temp')


import logging
import pandas as pd

### add to log module ###
logging.basicConfig(filename='app.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

def df_len(name, df):
    logging.info(f'len: {len(df)} \t\t {name}')

### import into main() ###
# import log.log as log
# log.df_len()

load = pd.DataFrame()
df_len('load', load)


from datetime import date, timedelta, datetime
import holidays

today = date.today()
HOLIDAYS_US = holidays.US(years= today.year)
HOLIDAYS_company = dict(zip(HOLIDAYS_US.values(), HOLIDAYS_US.keys()))

del_list = ("Washington\'s Birthday", 'Juneteenth National Independence Day','Columbus Day','Veterans Day')
for i in del_list:
    HOLIDAYS_company.pop(i)


ONE_DAY = timedelta(days=1)

def next_business_day(start):
    next_day = start + ONE_DAY
    while next_day.weekday() in holidays.WEEKEND or next_day in HOLIDAYS_company.values():
        next_day += ONE_DAY
    return next_daye

def last_business_day(start):
    next_day = start - ONE_DAY
    while next_day.weekday() in holidays.WEEKEND or next_day in HOLIDAYS_company.values():
        next_day -= ONE_DAY
    return next_day

def x_Bus_Day_ago(N):
    B10 = []
    seen = set(B10)
    i = today

    while len(B10) < N:
        item = last_business_day(i)
        if item not in seen:
            seen.add(item)
            B10.append(item)
        i -= timedelta(days=1)
    return B10[-1]


# I've found pandas.isin or set.intersection()
# Are very comparable

import pandas as pd
import numpy as np

df = pd.DataFrame()

def list_in_column(df, list):
    filter0 = df['col_name'].isin(list)
    df['new_col'] = np.where(filter0, 1, 0)
    return df

# pandas_df     = list_add['OutreachID'].squeeze()
# pandas_series = list_add['OutreachID'].tolist()


def match_column(df):
    df['OutreachID'] = df['OutreachID'].astype(str)
    df['Matches'] = df.groupby(['PhoneNumber'])['OutreachID'].transform(lambda x : '|'.join(x)).apply(lambda x: x[:3000])
    return df


import pyarrow as pa
import pyarrow.csv as csv

def data_proccess(location):
    z = Path(f'{location}')
    b = list(z.glob(f'*.csv'))

    final = pd.DataFrame()
    for i in b:
        table = csv.read_csv(i)
        df = table.to_pandas()
        ### get name of file
        # st = (str(i).split('\\')[-1][:-4])
        st = 'file_name'
        
        ### filter what you want
        # today = datetime.strptime(st, "%Y-%m-%d")
        # yesterday = last_business_day(today)
        filter1 = 'yesterday'
        filter2 = 'properly'

        ### create list
        pastdue = df[df.Outreach_Status == 'Past Due'][['OutreachID']]

        worked = df[df.Last_Call == filter1]

        worked_ls = worked['OutreachID'].tolist()

        worked_properly = worked[worked.Outreach_Status == filter2]
        worked_properly_ls = worked_properly['OutreachID'].tolist()
        try:
            ### try and skip first file, second file uses "last"
            print(st)
            total_work   = last[last.OutreachID.isin(worked_ls)]
            total_proper = last[last.OutreachID.isin(worked_properly_ls)]

            total        = len(last)
            work         = len(total_work)
            count        = len(total_proper)
            pct = count / work
            
            final[f'{st}'] = [total, work, count, pct]
            last = pastdue
        except:
            last = pastdue
    return final
final = data_proccess('temp')
final = final.T
final.columns = ['Total PastDue', 'Total Worked', 'Next Day Schedule', '%']


def pivot_tables(df):
    df.pivot_table(index =['Daily_Priority', 'Daily_Groups', 'rolled'], 
                    columns ='Skill', 
                    values ='PhoneNumber', 
                    aggfunc = ['count'], 
                    margins=True,
                    margins_name= 'TOTAL')


# and -> &
# or  -> | 
# not -> !=
# < , > , <= , >=

def np_filters(df, ifTrue):
    # filters
    f1 = df['col1'] == 'x'
    f2 = df['col2'] == 'y'
    # replace based on filer 
    # filter, if True, if False
    df['Skill'] = np.where(f1 | f2, ifTrue, df['Skill'])
    return df


### Input/output static tables ###
# supports module setup where you only need to add paths once
# for example ./src/pipeline/tables 
# holds all the sources code, while 
# ./data/table_drops holds the static table
# when I import this as a function into main everything is taken care of
from pathlib import Path
import os

paths = Path(__file__).parent.absolute().parent.absolute().parent.absolute()
table_path   = paths / "data/table_drop"

def tables(push_pull, table, name, path=table_path):
    if push_pull == 'pull':
        # return csv.read_csv(paths / path / name)
        return pd.read_csv(path / name, sep=',', on_bad_lines='warn', engine="python",)
    else:
        table.to_csv(table_path / name, sep=',', index=False)

start = tables('pull', 'NA', 'start.csv')
tables('push', start, 'start.csv')


import pandas as pd
import pyodbc

# ./src/server/queries/master_reporting.py
def sql(name):
	sql = (f"""
        SELECT *
        FROM db.Prod.Master_Reporting AS mr
        WHERE mr.names = '{name}'
	""")
	return sql

# ./src/server/query.py
def query(servername, database, sql, query_name):
      # create the connection
      try:
            conn = pyodbc.connect(f"""
                  DRIVER={{SQL Server}};
                  SERVER={servername};
                  DATABASE={database};
                  Trusted_Connection=yes""",
                  autocommit=True) 
      except pyodbc.OperationalError:
            print("""Couldn\'t connect to server""")
            query(servername, database, sql, query_name)
      else:
            print(f'''Connected to Server \t {query_name}''')
            df = pd.read_sql(sql, conn)
            return df

# ./src/main.py
MR_sql = server.queries.master_reporting.sql('aaron')
master_reporting   = server.query.query(servername, 
                                        database,
                                        MR_sql,
                                        'master_reporting')


### Using aggrigation, collect top n and convert the remainder into one group.

import pandas as pd

df = pd.DataFrame()

top = 10
ascend = False    # it can also be smallest items, set ascend = True 

group   = 'col_name'
agg_col = 'col_name'
agg     = 'agg_type' # count, mean, sum, median, min, max, mode, std, var

top_groups = df.groupby(group).agg({agg_col:agg}).sort_values(by=agg_col,ascending=ascend)
top_groups = top_groups[:top].copy()
top_groups.loc[f'Not Top {top}'] = top_groups[top:].sum()


### format dataframe, update format through { }
df = pd.DataFrame()
df.apply(lambda series: series.apply('{:.2%}'.format))


import pandas as pd
df = pd.DataFrame({
        'Skill':['a','a','a','b','b','b','a','a','a','b','b','b',],
        'meet_sla':[0,0,0,0,0,0,1,1,1,0,1,1],
        'has_call':[0,1,1,0,0,0,0,1,1,0,1,1],
        'age':[1,23,4,5,6,78,90,98,76,5,13,30],
    })

# rank a df with optional subgroups that interal sort  
def rank(df=pd.DataFrame, new_col=str, groups=list, rank_cols=dict):
    sort_columns = groups + [*rank_cols.keys()]
    ascending    = [True] * len(groups) + [*rank_cols.values()]
    
    df.sort_values(sort_columns, ascending=ascending, inplace=True)
    df[new_col] = 1
    df[new_col] = df.groupby(groups)[new_col].cumsum()
    return df.reset_index(drop=True)

new_col = 'overall_rank'
groups  = ['Skill']
rank_cols = {'meet_sla':True, 'has_call':True, 'age':False}
# group by phone number or msid & rank highest value org
rank(df, new_col, groups, rank_cols)


