select
    Review_Key,
    Review_ID,
    ASIN,
    Rating,
    Title,
    Content,
    Clean_Text,
    Review_Date,
    Verified,
    Helpful_Votes,
    Variant,
    Vine,
    Images,
    Videos,
    Source_File,
    ETL_Time

from {{ source('voc_raw', 'review_raw') }}