import pandas as pd

DATABASE = "loans_learning"

pd.options.display.max_columns = 500
pd.options.display.expand_frame_repr = False

monthlyDescriptionPath = "Mortgages_freddie.csv"
monthly = pd.read_csv(monthlyDescriptionPath, index_col="Column_name")
monthly = monthly[monthly["Affects_reposession"] > 0]   # Remove column names that shoudldn't have any effect on the prediction

monthlyColumnNames = monthly.index # Array of strings of column names that do have an effect on the prediction
monthlyColumnNames = ", ".join(monthlyColumnNames) + ", loan_sequence_number" + ", default_flag"  # For index and result

# co_borrower_credit_score, sato, hpi_at_origination are all NULL
import pg8000
conn = pg8000.connect(database="agency-loan-level", user="postgres", password="password")

def getDefaultData(amount="50"):
    query = "SELECT " + monthlyColumnNames + " FROM " + DATABASE + " WHERE default_flag IS TRUE AND random() < 0.01 LIMIT " + amount
    defaults = pd.read_sql_query(query, conn, index_col="loan_sequence_number", parse_dates=None)
    return defaults

def getNonDefaultData(amount="50"):
    query = "SELECT " + monthlyColumnNames + " FROM " + DATABASE + " WHERE default_flag IS FALSE AND random() < 0.01 LIMIT " + amount
    nonDefaults = pd.read_sql_query(query, conn, index_col="loan_sequence_number", parse_dates=None)
    return nonDefaults

def getData(amount="50"):
    query = "SELECT " + monthlyColumnNames + " FROM " + DATABASE + " WHERE random() < 0.01 LIMIT " + amount
    df = pd.read_sql_query(query, conn, index_col="loan_sequence_number", parse_dates=None)
    return df

