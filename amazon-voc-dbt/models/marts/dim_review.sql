select

    c.Review_ID,
    c.Clean_Text,
    c.Rating,
    c.Helpful_Votes as Helpful,
    c.Review_Date,
    f.Keywords

from {{ ref('int_canonical_reviews') }} c

left join {{ source('voc_features', 'review_features') }} f

on c.Review_ID = f.Review_ID