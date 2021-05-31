/*
    Churn Prediction Project Example

    We are going to query the granular thumbs up and down page views per user
    which are going be aggregated.

*/

SELECT userId,
       1                                    AS thumbsdown,
       to_timestamp(from_unixtime(ts/1000)) AS timestamp
FROM events
WHERE userId IS NOT NULL AND page = 'Thumbs Down'
