# Kaggle: Airbnb New User Bookings

This repository contains python script for [Kaggle: Airbnb New User Bookings](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings).


## My Results
### Final Submitted Version

- Public LB score: 0.88029 (134th / 1463)
- Private LB score: 0.88566 (86th / 1463)

### This Repository Version

I modified model after the compettion is over.

- Public LB score: 0.88097
- Private LB score: 0.88598


## Models

I used the xgboost to create following models.

|      Models      |  Public | Private |
|:----------------:|--------:|--------:|
| no sessions data | 0.86531 | 0.86991 |
| users + sessions | 0.88097 | 0.88598 |

Note: `users + sessions` Model requires 32GB+ RAM because `sessions.csv` has 10M+ rows.


## Dependencies

- Ubuntu 14.04
- Python 2.7.11 Anaconda 2.3.0 (64-bit)


### Python Packages

- numpy 1.10.4
- scipy 0.16.1
- pandas 0.16.2
- scikit-learn 0.17
- xgboost 0.4a29


## How to Run

Please download [data](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data) from kaggle
and add csv files into `./input/*.csv`.

```
$python all_run.py
```

Note: It takes around 5 hours on my PC(Intel Core i5, 32GB RAM).
