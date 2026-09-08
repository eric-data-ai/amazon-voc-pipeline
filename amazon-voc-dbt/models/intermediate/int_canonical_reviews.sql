WITH canonical_reviews AS (

    SELECT
        Review_ID,
        Clean_Text,
        Rating,
        Helpful_Votes,
        Review_Date,

        ROW_NUMBER() OVER (
            PARTITION BY Review_ID
            ORDER BY
                CASE
                    WHEN COALESCE(Content, '') <> '' THEN 1
                    ELSE 0
                END DESC,
                LENGTH(Content) DESC,
                LENGTH(Clean_Text) DESC
        ) AS rn

    FROM {{ ref('stg_review') }}

    WHERE Clean_Text <> ''
)

SELECT
    Review_ID,
    Clean_Text,
    Rating,
    Helpful_Votes,
    Review_Date

FROM canonical_reviews

WHERE rn = 1