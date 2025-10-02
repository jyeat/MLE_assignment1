import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_feature_gold_table(snapshot_date_str, silver_attributes_directory,silver_clickstream_directory,silver_financials_directory, gold_feature_store_directory,spark):
    #prepare arguments
    snapshot_date=datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to silver attribute and finance table
    attr_partition_name="silver_attributes_"+ snapshot_date_str.replace('-','_') + '.parquet'
    attr_filepath=silver_attributes_directory + attr_partition_name
    df_attr=spark.read.parquet(attr_filepath)

    fin_partition_name="silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    fin_filepath=silver_financials_directory +fin_partition_name
    df_fin=spark.read.parquet(fin_filepath)

    # Get last month clickstream data
    prev_month_date=snapshot_date-relativedelta(months=1)
    prev_month_str=prev_month_date.strftime("%Y_%m_%d")

    #Load previous month's clickstream file
    prev_click_partition_name=f"silver_clickstream_{prev_month_str}.parquet"
    prev_click_filepath=silver_clickstream_directory + prev_click_partition_name

    try:
        df_click=spark.read.parquet(prev_click_filepath)

        df_click_recent=df_click.groupBy("Customer_ID").agg(
            F.count('*').alias("visit_frequency"), #Counts number of visits
            F.max('snapshot_date').alias("last_activity_date"), #most recent visit
            F.sum('total_activity').alias("recent_total_activity"), #Sum of activities
            F.avg('activity_intensity').alias("avg_activity_intensity"), #Avg of intensities
            F.sum('active_features_count').alias("total_active_features"), #Sum of active features
        )
    
    except:
        #fallback if no previous month clickstream data
        print(f"Warning: No clickstream data found for previous month {prev_month_str}")
        # Create empty dataframe with correct schema
        from pyspark.sql.types import StructType, StructField
        schema = StructType([
            StructField("Customer_ID", StringType(), True),
            StructField("visit_frequency", IntegerType(), True),
            StructField("last_activity_date", DateType(), True),
            StructField("recent_total_activity", FloatType(), True),
            StructField("avg_activity_intensity", FloatType(), True),
            StructField("total_active_features", IntegerType(), True)
        ])
        df_click_recent = spark.createDataFrame([], schema)

    #join table
    feature_df = df_attr.join(df_fin, on=["Customer_ID", "snapshot_date"], how="left")
    feature_df = feature_df.join(df_click_recent, on=["Customer_ID"], how="left")

    #aggregate    
    #financial health
    feature_df = feature_df.withColumn("financial_health_score",
                                       (F.when(F.col("Debt_to_income_ratio") < 0.3, 3)
                                        .when(F.col("Debt_to_income_ratio") < 0.5, 2)
                                        .otherwise(1)) +
                                        (F.when(F.col("Credit_Utilization_Ratio") < 30, 2)
                                         .when(F.col("Credit_Utilization_Ratio") < 70, 1)
                                         .otherwise(0)))
    
    
    #drop unnessary columns
    feature_df=feature_df.drop("Name","SSN","SSN_is_valid","SSN_clean",'Occupation', 'age_group', #attributes
                               "Monthly_Inhand_Salary", "Credit_History_Age","credit_risk_level") #finance

    
    # save gold table 
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    feature_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return feature_df

def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df