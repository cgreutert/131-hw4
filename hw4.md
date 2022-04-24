Homework Four: Resampling
================
Carly Greutert
24 April 2022

``` r
titanic <- read_csv('C:\\Program Files\\Git\\tmp\\131-hw4\\titanic.csv')
names <- c('pclass', 'survived')
titanic[,names] <- lapply(titanic[,names] , factor)
```

1.  

``` r
set.seed(777)
titanic_split <- initial_split(titanic, prop = 0.80, strata = 'survived')
titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)
titanic_recipe <- recipe(survived ~ pclass + sex + age + sib_sp + parch + fare, data = titanic_train)
titanic_recipe <- titanic_recipe %>%
  step_impute_linear(age) %>%
  step_dummy(all_nominal_predictors())%>%
  step_interact(~sex:fare) %>%
  step_interact(~age:fare)
dim(titanic_train)
```

    ## [1] 712  12

``` r
dim(titanic_test)
```

    ## [1] 179  12

From the dim() function above, we can verify the number of observations
was split correctly.  
2.

``` r
titanic_folds <- vfold_cv(titanic_train, v = 10)
titanic_folds
```

    ## #  10-fold cross-validation 
    ## # A tibble: 10 x 2
    ##    splits           id    
    ##    <list>           <chr> 
    ##  1 <split [640/72]> Fold01
    ##  2 <split [640/72]> Fold02
    ##  3 <split [641/71]> Fold03
    ##  4 <split [641/71]> Fold04
    ##  5 <split [641/71]> Fold05
    ##  6 <split [641/71]> Fold06
    ##  7 <split [641/71]> Fold07
    ##  8 <split [641/71]> Fold08
    ##  9 <split [641/71]> Fold09
    ## 10 <split [641/71]> Fold10

What we are doing above is resampling by splitting our training data
into further subsets in order to provide a better evaluation for our
model. K-fold cross-validation splits our data K-ways and has K-1
subsets and one subset for validation. It will help us train our model
further. We should use k-fold cross-validation, rather than simply
fitting and testing models on the entire training set in order to decide
which model produces the lowest error rates and is therefore the best
model to use. If we did use the entire training set, the resampling
method used would be the validation set approach.
