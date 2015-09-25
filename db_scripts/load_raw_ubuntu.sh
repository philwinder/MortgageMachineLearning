#!/bin/bash

# download raw data from fannie mae and freddie mac's websites:
# http://www.freddiemac.com/news/finance/sf_loanlevel_dataset.html

# Unzip all files into seperate directories with (osx and ubuntu):
# find *.zip -exec sh -c 'unzip -d "${1%.*}" "$1"' _ {} \;

# Note (osx and ubuntu): Delete old folders with
# find his* -type d -exec sh -c 'rm -rf "$1"' _ {} \;
# Note (osx): Delete old zips with
# find his*.zip -type f -exec sh -c 'rm -rf "$1"' _ {} \;
# Note (ubuntu): commands require sudo, and create db must be used in the name of the user. e.g.
# sudo -u postgres createdb agency-loan-level root

# ubuntu postgres users may need to run the following to add a default password:
# ALTER USER postgres PASSWORD 'my_postgres_password';


app_dir="${PWD}/"
freddie_script="populate_freddie_from_raw.sql"
base_data_dir="/home/phil/work/philwinder/MortgageMachineLearning/raw/"

echo "Creating DB"
# Create db
sudo -u postgres createdb agency-loan-level root
# Create tables
echo "Creating tables"
sudo -u postgres psql -d agency-loan-level -f create_tables.sql

# Find data and insert (This will take a long time¬
echo "Importing data"
for y in {2013..1999}
do
echo "$y"
  for q in {4..1}
  do
    echo "$q"
    if (($y < 2013 || $q < 4)); then
      echo "`date`: beginning freddie load for $y Q$q"

      freddie_loans_file="historical_data1_Q$q$y/historical_data1_Q$q$y.txt"
      cat $base_data_dir$freddie_loans_file | sudo -u postgres psql agency-loan-level -c "COPY loans_raw_freddie FROM stdin DELIMITER '|' NULL '';"
      echo "`date`: loaded freddie raw loans for $y Q$q"

      freddie_monthly_file="historical_data1_Q$q$y/historical_data1_time_Q$q$y.txt"
      cat $base_data_dir$freddie_monthly_file | sudo -u postgres psql agency-loan-level -c "COPY monthly_observations_raw_freddie FROM stdin DELIMITER '|' NULL '';"
      echo "`date`: loaded freddie raw monthly observations for $y Q$q"

      # sudo -u postgres psql agency-loan-level -f $app_dir$freddie_script
      echo "`date`: finished freddie loans and monthly observations for $y Q$q"
    fi
  done
done

# Clean raw data and populate new table
sudo -u postgres psql -d agency-loan-level -f InsertCleanData.sql;