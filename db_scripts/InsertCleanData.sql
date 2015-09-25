DROP TABLE IF EXISTS loans_learning;
CREATE TABLE loans_learning (
  id integer NOT NULL,
  first_payment_date date,
  credit_score integer NOT NULL,
  first_time_homebuyer_flag integer NOT NULL,
  mip integer,
  number_of_units integer,
  occupancy_status integer NOT NULL,
  ocltv numeric,
  dti integer NOT NULL,
  original_upb numeric,
  oltv numeric,
  original_interest_rate numeric,
  channel integer NOT NULL,
  prepayment_penalty_flag integer NOT NULL,
  property_type integer NOT NULL,
--  postal_code integer,
  loan_sequence_number char(12),
  loan_purpose integer NOT NULL,
  original_loan_term integer,
  number_of_borrowers integer NOT NULL,
  hpi_at_origination numeric,
  default_flag boolean
);

--CREATE SEQUENCE loans_id_seq
--  START WITH 1
--  INCREMENT BY 1
--  NO MINVALUE
--  NO MAXVALUE
--  CACHE 1;

ALTER TABLE ONLY loans_learning ALTER COLUMN id SET DEFAULT nextval('loans_id_seq'::regclass);
ALTER TABLE ONLY loans_learning ADD CONSTRAINT loans_learning_pkey PRIMARY KEY (id);
CREATE UNIQUE INDEX index_loans_learning_on_seq ON loans_learning (loan_sequence_number);

DROP TABLE IF EXISTS hpi_indexes;
CREATE TABLE hpi_indexes (
  id integer PRIMARY KEY,
  name varchar,
  type varchar,
  first_date date
);

COPY hpi_indexes FROM '/Volumes/source/FreddieMac/sql/hpi_index_codes.txt' DELIMITER '|' NULL '';

DROP TABLE IF EXISTS hpi_values;
CREATE TABLE hpi_values (
  hpi_index_id integer,
  date date,
  hpi numeric,
  PRIMARY KEY (hpi_index_id, date)
);

COPY hpi_values FROM '/Volumes/source/FreddieMac/sql/interpolated_hpi_values.txt' DELIMITER '|' NULL '';


INSERT INTO loans_learning
  (credit_score, first_payment_date, first_time_homebuyer_flag, mip, number_of_units,
    occupancy_status, ocltv, dti, original_upb, oltv, original_interest_rate, channel, prepayment_penalty_flag,
    property_type,
--    postal_code,
    loan_sequence_number, loan_purpose,
    original_loan_term, number_of_borrowers, hpi_at_origination, default_flag)
SELECT
  (CASE
    WHEN credit_score='   ' THEN '600' --Three spaces, if Credit Score is < 301 or > 850.
    WHEN credit_score='' THEN '0' -- Will try and fill data
    WHEN (credit_score)::integer > 0 THEN credit_score
    ELSE '300' -- When null, is risky
  END)::integer,
  (first_payment_date || '01')::date,
  (CASE
    WHEN first_time_homebuyer_flag='N' THEN 1
    WHEN first_time_homebuyer_flag='Y' THEN 2
    ELSE 0
  END),
  NULLIF(mip, '')::integer,
  number_of_units,
  (CASE
    WHEN occupancy_status='O' THEN 1
    WHEN occupancy_status='I' THEN 2
    WHEN occupancy_status='S' THEN 3
    ELSE 0
  END),
  ocltv,
  (CASE
    WHEN dti='   ' THEN '65' -- When three spaces is > 65
    WHEN dti='' THEN '0' -- Will try and fill data
    WHEN (dti)::integer > 0 THEN dti
    ELSE '0' -- When null, is risky, so assume blank
  END)::integer,
  original_upb, oltv, original_interest_rate,
  (CASE
    WHEN channel='R' THEN 1
    WHEN channel='B' THEN 2
    WHEN channel='C' THEN 3
    WHEN channel='T' THEN 4
    ELSE 0
  END),
  (CASE
    WHEN prepayment_penalty_flag='N' THEN 1
    WHEN prepayment_penalty_flag='Y' THEN 2
    ELSE 0
  END),
  (CASE
    WHEN property_type='CO' THEN 1
    WHEN property_type='LH' THEN 2
    WHEN property_type='PU' THEN 3
    WHEN property_type='MH' THEN 4
    WHEN property_type='SF' THEN 5
    WHEN property_type='CP' THEN 6
    ELSE 0
  END),
--  NULLIF(postal_code, '')::integer,
  raw.loan_sequence_number,
  (CASE
    WHEN loan_purpose='P' THEN 1
    WHEN loan_purpose='C' THEN 2
    WHEN loan_purpose='N' THEN 3
    ELSE 0
  END),
  original_loan_term,
  (CASE
    WHEN number_of_borrowers IS NULL THEN '0' -- When two spaces, unknown
    WHEN number_of_borrowers > 0 THEN number_of_borrowers
    ELSE '0'
  END)::integer,
  hpi.hpi,
  (CASE
    WHEN md.first_serious_dq_date IS NULL THEN FALSE
    ELSE TRUE
  END)::boolean
FROM loans_raw_freddie raw
  LEFT JOIN hpi_indexes hpi_msa
    ON raw.msa = hpi_msa.id AND ((first_payment_date || '01')::date - interval '1 month') > hpi_msa.first_date
  LEFT JOIN hpi_indexes hpi_state
    ON raw.property_state = hpi_state.name
  LEFT JOIN hpi_values hpi
    ON hpi.hpi_index_id = COALESCE(COALESCE(hpi_msa.id, hpi_state.id), 0) -- Sets HPI to national average if null
    AND hpi.date = ((first_payment_date || '01')::date - interval '2 months')
  LEFT JOIN (SELECT
               loan_sequence_number,
               reporting_period,
               zero_balance_code,
               zero_balance_effective_date,
               net_sales_proceeds,
               mi_recoveries,
               non_mi_recoveries,
               expenses,
               ROW_NUMBER() OVER (PARTITION BY loan_sequence_number ORDER BY reporting_period ASC) AS row_num
             FROM monthly_observations_raw_freddie
             WHERE zero_balance_code IS NOT NULL) mz
    ON raw.loan_sequence_number = mz.loan_sequence_number
    AND mz.row_num = 1
  LEFT JOIN (SELECT
               loan_sequence_number,
               MIN(reporting_period) AS first_serious_dq_date
             FROM monthly_observations_raw_freddie
             WHERE dq_status NOT IN ('0', '1')
             GROUP BY loan_sequence_number) md
    ON raw.loan_sequence_number = md.loan_sequence_number;


-- Note, add in mortgage rates again
-- Node, add in postal code somehow. E.g. average worth in postcode.

    --
-- id                        | integer              | not null default nextval('loans_id_seq'::regclass)
-- agency                    | integer              |
-- credit_score              | integer              |
-- first_payment_date        | date                 |
-- first_time_homebuyer_flag | character(1)         |
-- maturity_date             | date                 |
-- msa                       | integer              |
-- mip                       | integer              |
-- number_of_units           | integer              |
-- occupancy_status          | character(1)         |
-- ocltv                     | numeric              |
-- dti                       | integer              |
-- original_upb              | numeric              |
-- oltv                      | numeric              |
-- original_interest_rate    | numeric              |
-- channel                   | character(1)         |
-- prepayment_penalty_flag   | character(1)         |
-- product_type              | character varying(5) |
-- property_state            | character(2)         |
-- property_type             | character(2)         |
-- postal_code               | integer              |
-- loan_sequence_number      | character(12)        |
-- loan_purpose              | character(1)         |
-- original_loan_term        | integer              |
-- number_of_borrowers       | integer              |
-- seller_id                 | integer              |
-- servicer_id               | integer              |
-- vintage                   | integer              |
-- hpi_index_id              | integer              |
-- hpi_at_origination        | numeric              |
-- final_zero_balance_code   | integer              |
-- final_zero_balance_date   | date                 |
-- first_serious_dq_date     | date                 |
-- sato                      | numeric              |
-- mi_recoveries             | numeric              |
-- net_sales_proceeds        | numeric              |
-- non_mi_recoveries         | numeric              |
-- expenses                  | numeric              |
-- co_borrower_credit_score  | integer              |
--Indexes:
--    "loans_pkey" PRIMARY KEY, btree (id)
--    "index_loans_on_seq" UNIQUE, btree (loan_sequence_number, agency)
--