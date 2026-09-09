select distinct

    ASIN,
    Review_ID,
    Review_Key

from {{ ref('stg_review') }}