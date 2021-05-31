/*
    Churn Prediction Project Example

    Churn flag per user

*/

SELECT distinct(e.userId), COALESCE(p.is_churned, 0) AS is_churned
FROM events e
LEFT JOIN (SELECT userId, 1 AS is_churned
            FROM events
            WHERE page = 'Cancellation Confirmation') p
    ON (e.userId = p.userId)
WHERE e.userId IS NOT NULL
