#!/bin/bash

# download raw data from fannie mae and freddie mac's websites:
# http://www.freddiemac.com/news/finance/sf_loanlevel_dataset.html

# Unzip all files into seperate directories with:
# find *.zip -exec sh -c 'unzip -d "${1%.*}" "$1"' _ {} \;

# Note (osx): Delete old folders with
# find his* -type d -exec sh -c 'rm -rf "$1"' _ {} \;
# Note (osx): Delete old zips with
# find his*.zip -type f -exec sh -c 'rm -rf "$1"' _ {} \;

app_dir="${PWD}/"
freddie_script="populate_freddie_from_raw.sql"
base_data_dir="/tmp/freddie/"

# Create db
createdb agency-loan-level
# Create tables
psql -d agency-loan-level -f create_tables.sql

# Find data and insert
for y in $(seq 2013 1999)
do
  for q in $(seq 4 1)
  do
    if (($y < 2013 || $q < 4)); then
      echo "`date`: beginning freddie load for $y Q$q"

      freddie_loans_file="historical_data1_Q$q$y/historical_data1_Q$q$y.txt"
      cat $base_data_dir$freddie_loans_file | psql agency-loan-level -c "COPY loans_raw_freddie FROM stdin DELIMITER '|' NULL '';"
      echo "`date`: loaded freddie raw loans for $y Q$q"

      freddie_monthly_file="historical_data1_Q$q$y/historical_data1_time_Q$q$y.txt"
      cat $base_data_dir$freddie_monthly_file | psql agency-loan-level -c "COPY monthly_observations_raw_freddie FROM stdin DELIMITER '|' NULL '';"
      echo "`date`: loaded freddie raw monthly observations for $y Q$q"

      psql agency-loan-level -f $app_dir$freddie_script
      echo "`date`: finished freddie loans and monthly observations for $y Q$q"
    fi
  done
done

# Clean raw data and populate new table
psql -d agency-loan-level -f InsertCleanData.sql;