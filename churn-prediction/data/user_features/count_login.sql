/*
    Churn Prediction Project Example

    We are going to query the granular login data per user which is going be
    used to be aggregated into two features:

    1. login_count_7d
    2. login_count_30d

*/

SELECT userId,
       1                                            AS login,
       first(to_timestamp(from_unixtime(ts/1000)))  AS timestamp
FROM events
WHERE userId IS NOT NULL
GROUP BY userId, sessionId
