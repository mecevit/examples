/*
    Churn Prediction Project Example

    Gender feature of the users

*/

SELECT userId,
    IF (gender = "F", 1, 0) AS gender
FROM events
WHERE userId IS NOT NULL
