/*
    Churn Prediction Project Example

    We are going to query the granular error page views per user which is going be
    used to be aggregated into three features:

    1. error_count_1d
    2. error_count_7d
    3. error_count_30d

*/

SELECT userId,
       1                                    AS error,
       to_timestamp(from_unixtime(ts/1000)) AS timestamp
FROM events
WHERE userId IS NOT NULL AND page = 'Error'
