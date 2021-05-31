/*
    Churn Prediction Project Example

    We are going to query the granular help page views per user which is going be
    used to be aggregated into three features:

    1. help_view_count_1d
    2. help_view_count_7d
    3. help_view_count_30d

*/

SELECT userId,
       1                                    AS help_view,
       to_timestamp(from_unixtime(ts/1000)) AS timestamp
FROM events
WHERE userId IS NOT NULL AND page = 'Help'
