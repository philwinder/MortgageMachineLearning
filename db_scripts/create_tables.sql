CREATE TABLE IF NOT EXISTS loans_raw_freddie (
  credit_score char(3), -- make into integer
  first_payment_date integer, -- make into date
  first_time_homebuyer_flag char(1),
  maturity_date integer, -- make into date
  msa integer,
  mip char(3),
  number_of_units integer,
  occupancy_status char(1),
  ocltv numeric,
  dti char(3),
  original_upb numeric,
  oltv integer,
  original_interest_rate numeric,
  channel char(1),
  prepayment_penalty_flag char(1),
  product_type char(5),
  property_state char(2),
  property_type char(2),
  postal_code char(5), -- make into integer
  loan_sequence_number char(12),
  loan_purpose char(1),
  original_loan_term integer,
  number_of_borrowers integer,
  seller_name varchar(30),
  servicer_name varchar(30)
);

CREATE TABLE IF NOT EXISTS loans (
  id integer NOT NULL,
  agency integer,
  credit_score integer,
  first_payment_date date,
  first_time_homebuyer_flag char(1),
  maturity_date date,
  msa integer,
  mip integer,
  number_of_units integer,
  occupancy_status char(1),
  ocltv numeric,
  dti integer,
  original_upb numeric,
  oltv numeric,
  original_interest_rate numeric,
  channel char(1),
  prepayment_penalty_flag char(1),
  product_type varchar(5),
  property_state char(2),
  property_type char(2),
  postal_code integer,
  loan_sequence_number char(12),
  loan_purpose char(1),
  original_loan_term integer,
  number_of_borrowers integer,
  seller_id integer,
  servicer_id integer,
  vintage integer,
  hpi_index_id integer,
  hpi_at_origination numeric,
  final_zero_balance_code integer,
  final_zero_balance_date date,
  first_serious_dq_date date,
  sato numeric,
  mi_recoveries numeric,
  net_sales_proceeds numeric,
  non_mi_recoveries numeric,
  expenses numeric,
  co_borrower_credit_score integer
);

CREATE SEQUENCE loans_id_seq
  START WITH 1
  INCREMENT BY 1
  NO MINVALUE
  NO MAXVALUE
  CACHE 1;

ALTER TABLE ONLY loans ALTER COLUMN id SET DEFAULT nextval('loans_id_seq'::regclass);
ALTER TABLE ONLY loans ADD CONSTRAINT loans_pkey PRIMARY KEY (id);

CREATE UNIQUE INDEX index_loans_on_seq ON loans (loan_sequence_number, agency);

CREATE TABLE IF NOT EXISTS servicers (
  id integer NOT NULL,
  name varchar(80)
);

CREATE UNIQUE INDEX index_servicers_on_name ON servicers (name);

CREATE SEQUENCE servicers_id_seq
  START WITH 1
  INCREMENT BY 1
  NO MINVALUE
  NO MAXVALUE
  CACHE 1;

ALTER TABLE ONLY servicers ALTER COLUMN id SET DEFAULT nextval('servicers_id_seq'::regclass);
ALTER TABLE ONLY servicers ADD CONSTRAINT servicers_pkey PRIMARY KEY (id);

CREATE TABLE IF NOT EXISTS  monthly_observations_raw_freddie (
  loan_sequence_number char(12), -- replace with integer loan id
  reporting_period integer, -- make into date
  current_upb numeric,
  dq_status varchar(3), -- make into integer
  loan_age integer,
  rmm integer,
  repurchase_flag char(1),
  modification_flag char(1),
  zero_balance_code char(2), -- make into integer
  zero_balance_effective_date integer, -- make into date
  current_interest_rate numeric,
  current_deferred_upb numeric,
  ddlpi integer,
  mi_recoveries numeric,
  net_sales_proceeds varchar(20),
  non_mi_recoveries numeric,
  expenses numeric
);

CREATE TABLE IF NOT EXISTS  monthly_observations (
  loan_id integer,
  date date,
  current_upb numeric,
  previous_upb numeric,
  dq_status integer,
  previous_dq_status integer,
  loan_age integer,
  rmm integer,
  repurchase_flag char(1),
  modification_flag char(1),
  zero_balance_code integer,
  zero_balance_date date,
  current_interest_rate numeric
);

DROP VIEW loan_monthly;
CREATE VIEW loan_monthly AS
SELECT
  l.*,
  m.loan_id, m.date, m.current_upb, m.previous_upb, m.dq_status, m.previous_dq_status,
  m.loan_age, m.rmm, m.repurchase_flag, m.modification_flag, m.current_interest_rate, m.zero_balance_code,
  COALESCE(m.current_upb, l.original_upb) AS current_weight,
  COALESCE(m.previous_upb, l.original_upb) AS previous_weight
FROM loans l
  INNER JOIN monthly_observations m
    ON l.id = m.loan_id;

CREATE OR REPLACE FUNCTION cpr(numeric) RETURNS numeric
  AS 'SELECT (1.0 - pow(1.0 - $1, 12)) * 100;'
  LANGUAGE SQL
  IMMUTABLE
  RETURNS NULL ON NULL INPUT;

DROP TABLE IF EXISTS mortgage_rates;
CREATE TABLE mortgage_rates (
  month date PRIMARY KEY,
  rate numeric,
  points numeric,
  zero_point_rate numeric
);

COPY mortgage_rates FROM '/home/phil/work/philwinder/MortgageMachineLearning/db_scripts/pmms.csv' DELIMITER ',' NULL '';